-- Steering the scorecard: one round of meta-cognition.
--
-- Tactus owns this loop. The deterministic work -- reading the feedback, fitting the head,
-- pricing a Jev top-up, committing a new scorecard version -- lives in the Python host
-- module registered as "flywheel". The parts that need judgement are here: asking a
-- language model why the predictions are wrong, deciding whether its proposal is worth
-- paying to evaluate, and putting a human in front of anything that changes the scorecard.
--
-- The rules this procedure encodes, each of which is enforced by structure rather than by
-- asking nicely:
--
--   1. ONE proposal per round. Any change to the question set forces a fresh Jev pass over
--      the labeled items, so the agent gets a single batch of edits, never a loop.
--   2. The agent proposes EDITS, not numbers. The proposal format has no place for weights;
--      they come only from the deterministic fit.
--   3. Nothing costs money before it is priced, and nothing is applied before a human
--      approves it.
--   4. A candidate must beat the incumbent out of fold to be considered at all.
--
-- {{PROVIDER}}, {{MODEL}} and {{MAX_TOKENS}} are substituted by Python before execution.
--
-- NOTE: this procedure deliberately declares no `output` schema. An agent with no schema of
-- its own inherits the procedure's, and this agent replies in plain JSON text that the host
-- parses, which works on any model. Kimi K3 on Bedrock, for one, ignores the structured
-- output protocol and answers in prose.

local flywheel = require("flywheel")

analyst = Agent {
    provider = "{{PROVIDER}}",
    model = "{{MODEL}}",
    max_tokens = {{MAX_TOKENS}},
    system_prompt = [[
You are the steering analyst for a text-scoring system. A small model makes the final
decision from the answers to a set of yes/no, choice and score QUESTIONS, called ELEMENTS,
that an AI model answers about each text. Humans review its predictions one at a time and
sometimes explain why it was wrong. Your job is to read those disagreements and work out
which ELEMENT IS MISSING, WORDED BADLY, or USELESS, and to propose a small set of edits.

You will be given: the current scorecard, counts (including a FEATURE BUDGET), the
predictions the human disagreed with along with their comments, and an inventory of how much
each current element matters.

How to think:
- Look at BOTH lists. The disagreements show what is going wrong; the labeled sample shows
  what decides the label in the first place. Some factors are invisible in the disagreements
  precisely because the scorecard already handles them -- look for what the positive and
  negative examples have in common as groups, including things that are not about sentiment
  at all, such as what the text is about.
- Look for a pattern across the disagreements, not for a fix to each one. A comment that
  names a concept the current elements do not capture (sarcasm, hedging, a specific topic)
  is a missing element. A wrong prediction whose element answers were all confident and
  wrong suggests an element is worded ambiguously.
- Use permutation_importance to decide what to keep. An element near zero is not helping.
  Ignore raw weights; they are on different scales and mislead.
- Do not propose an element that merely restates one that exists.
- Prefer adding or rewording ELEMENTS. Rewording the holistic (top-level) question makes
  every stored answer to it stale for every item, which costs thousands of requests to
  refresh, and it stops the comparison with the original question meaning anything. Only
  do it if an element cannot capture the problem.
- Each element is one question answered in the same single request, so writing one is cheap.
  But every new element spends part of the feature budget, and changing the question set
  costs a fresh pass over the labeled items. You get ONE proposal this round. Make it count.
- Write each question so that it could be answered from the text alone, in one sentence,
  without needing to see the other questions.
- If the evidence does not support any change, propose nothing and say so. Do not change
  things just to have something to report.

Reply with ONE JSON object and nothing else, in exactly this form:

{
  "root_cause": "two or three sentences: the pattern you found and why it causes the errors",
  "add_elements": [
    {"key": "short_snake_case", "question_type": "noul", "instructions": "the question",
     "criteria": null}
  ],
  "retire_elements": ["key"],
  "reword_elements": [{"key": "existing_key", "instructions": "the new question"}]
}

question_type is "noul" (yes/no, no criteria), "choice" (criteria is a list of option names)
or "score" (criteria is an ordered list of levels). Keys use only letters, digits and
underscores. Use empty lists when there is nothing to change. Never include weights,
coefficients or numbers of any kind: they are set by a fit, not by you.
]],
}

-- A second analyst that is never told what the task is. It sees two groups of texts and is
-- asked what separates them. The framing matters: the analyst that can see the scorecard
-- reliably proposes refinements of the criterion it was shown, which is how a factor
-- orthogonal to that criterion stays invisible no matter how much evidence you add. This one
-- has no criterion to be loyal to.
scout = Agent {
    provider = "{{PROVIDER}}",
    model = "{{MODEL}}",
    max_tokens = {{MAX_TOKENS}},
    system_prompt = [[
You are shown two groups of short texts, Group A and Group B. They were sorted by a rule you
do not know. Your job is to work out what the rule might be.

Look at the groups as wholes and ask what the members of one have in common that the members
of the other do not. Consider ANY property: what the texts are about, their subject matter or
setting, who or what they describe, their tone, their structure, their vocabulary, how
strongly or faintly they put things, how long they are. Do not assume the rule is about any
one kind of property, and do not assume it is the most obvious one -- a rule that is obvious
from a handful of examples is usually not the rule that separates the whole set.

Propose up to five questions that would best separate the groups. Each must be answerable
from a single text on its own, by someone who has not seen the groups.

Reply with ONE JSON object and nothing else:

{
  "observations": "what you noticed about the two groups, in two or three sentences",
  "add_elements": [
    {"key": "short_snake_case", "question_type": "noul", "instructions": "the question",
     "criteria": null}
  ]
}

question_type is "noul" (yes/no, criteria null), "choice" (criteria is a list of options) or
"score" (criteria is an ordered list of levels). Propose several genuinely different
questions rather than five variations on one idea: they cost nothing to try, and the ones
that do not help will be discarded by measurement, not by argument.
]],
}

-- The briefing as text for the model. Kept in Lua, where it can be read and changed.
local function briefing_prompt(brief)
    return "CURRENT SCORECARD\n" .. brief.scorecard_yaml
        .. "\n\nCOUNTS AND BUDGET\n" .. Json.encode(brief.summary)
        .. "\n\nPREDICTIONS THE HUMAN DISAGREED WITH (commented ones first)\n"
        .. Json.encode(brief.mismatches)
        .. "\n\nCOMMENTS ON PREDICTIONS THE HUMAN AGREED WITH\n"
        .. Json.encode(brief.commented_agreements)
        .. "\n\nA SAMPLE OF LABELED ITEMS, RIGHT OR WRONG, BALANCED BY LABEL. Read these for "
        .. "regularities the error list cannot show: anything that decides the label but that "
        .. "the scorecard already gets right produces no errors to look at.\n"
        .. Json.encode(brief.labeled_sample)
        .. "\n\nHOW MUCH EACH ELEMENT MATTERS (permutation_importance is what to trust)\n"
        .. Json.encode(brief.element_inventory)
        .. "\n\nPropose your one round of edits now."
end

local TAXONOMY = [[

Kinds of convention worth considering, as a checklist, not a hint: what counts as in scope;
exceptions the labelers honour; the subject matter or setting of the text; its register or
formality; where a threshold sits between one label and the next.]]

-- An agent reply is plain text, or already-parsed data on some models. Hand the host a string.
local function as_text(reply)
    local out = reply.output
    if type(out) == "string" then
        return out
    end
    return Json.encode(out)
end

Procedure {
    input = {
        -- Evaluating a candidate that needs answers Jev has not given yet spends requests.
        -- Up to this many are approved automatically; more than that asks the human first.
        max_auto_requests = field.number{default = 300},
        -- How many times the agent may repair a proposal the host rejected. Repairing
        -- costs nothing (nothing has been fit or asked of Jev), and is not a second
        -- proposal: the question set has not changed.
        max_revisions = field.number{default = 1},
        -- Run the blind pass as well as the error analysis.
        discovery = field.boolean{default = false},
        -- Offer the error analyst a taxonomy of convention kinds. Recorded as its own arm
        -- because naming the kinds is a nudge, and a nudge has to be disclosed.
        taxonomy = field.boolean{default = false},
    },
    function(input)
        local brief = flywheel.briefing()
        local prompt = briefing_prompt(brief)
        Log.info("Steering round started", {labeled = brief.summary.n_labeled})

        -- 0. Optionally, look at the labels with no idea what the task is.
        local scout_reply = nil
        if input.discovery then
            scout_reply = as_text(scout({message =
                "Group A:\n" .. Json.encode(brief.blind_sample["Group A"])
                .. "\n\nGroup B:\n" .. Json.encode(brief.blind_sample["Group B"])
                .. "\n\nWhat separates these groups? Propose your questions now."}))
            Log.info("Blind pass complete")
        end

        -- 1. The error analysis, repaired at most `max_revisions` times if the host rejects it.
        local reply = analyst({message = prompt .. (input.taxonomy and TAXONOMY or "")})
        -- Candidates from both passes are pooled, not filtered by either: they ride in one Jev
        -- request, so trying five costs what trying one costs, and the fit decides.
        local check = flywheel.check_combined(as_text(reply), scout_reply)
        local revisions = 0
        while (not check.ok) and revisions < input.max_revisions do
            revisions = revisions + 1
            Log.info("Proposal rejected; asking for a repair", {problem = check.problems[1]})
            reply = analyst({message = prompt
                .. "\n\nYOUR PREVIOUS PROPOSAL WAS REJECTED:\n" .. check.problems[1]
                .. "\n\nReply again with one corrected JSON object."})
            check = flywheel.check_combined(as_text(reply), scout_reply)
        end
        if not check.ok then
            return {decision = "invalid_proposal", problem = check.problems[1]}
        end

        -- 2. Nothing to do is a legitimate answer, and the cheapest one.
        if check.noop then
            return {decision = "no_change_proposed", root_cause = check.root_cause}
        end

        -- 3. Price it before spending anything.
        if check.plan.requests > input.max_auto_requests then
            local go = Human.approve({message = "This proposal needs " .. tostring(check.plan.requests)
                .. " Jev requests (about " .. tostring(check.plan.estimated_input_tokens)
                .. " input tokens) to evaluate.\n\n" .. check.summary_text .. "\n\nSpend them?"})
            if not go then
                return {decision = "declined_spend", root_cause = check.root_cause}
            end
        end

        -- 4. Fit the candidate out of fold and compare it with the incumbent. This is the
        --    only call that spends, and it is made exactly once.
        local evaluation = flywheel.evaluate()
        if evaluation.status == "needs_spend" then
            return {decision = "needs_spend", requests = evaluation.requests,
                    root_cause = check.root_cause}
        end
        if evaluation.status ~= "fitted" then
            return {decision = "not_evaluable", status = evaluation.status, reason = evaluation.reason,
                    root_cause = check.root_cause}
        end
        if not evaluation.promote then
            return {decision = "rejected_by_metrics", reasons = evaluation.summary_text,
                    root_cause = check.root_cause}
        end

        -- 5. A human sees exactly what would change, and what it is projected to do.
        local approved = Human.approve({message = "Steering proposal\n\nWhy: " .. check.root_cause
            .. "\n\n" .. check.summary_text .. "\n\n" .. evaluation.summary_text
            .. "\n\nApply it as a new scorecard version?"})
        if not approved then
            return {decision = "rejected_by_human", root_cause = check.root_cause}
        end

        -- 6. Checkpointed, so a resumed run never applies it twice.
        local applied = Step.checkpoint(function()
            return flywheel.apply({model = "{{MODEL}}", revisions = revisions})
        end)
        return {decision = "promoted", version = applied.version, root_cause = check.root_cause}
    end
}

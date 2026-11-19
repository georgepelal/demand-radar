"""Regex signals for unmet needs, frustration and willingness to pay (English text)."""
from __future__ import annotations

import re

_NOUNS = (
    r"(?:tools?|apps?|applications?|services?|software|saas|products?|platforms?|websites?|sites?"
    r"|librar(?:y|ies)|solutions?|alternatives?|options?|programs?|startups?|compan(?:y|ies)|way)"
)
_GAP = r"(?:\W+\w+){0,3}?\W+"  # up to 3 words in between, e.g. "is there a *good free* tool"

PAIN_PATTERNS = {
    "wish_existed": r"\bi wish (?:there (?:was|were|is)|someone would|somebody would|(?:\w+\W+){0,8}?existed)\b",
    "someone_build": r"\b(?:someone|somebody) (?:should|needs? to|ought to|please) (?:build|make|create|write|start)\b"
                     r"|\bwish someone would build\b",
    "why_missing": r"\bwhy (?:is there no|isn't there|is there not|doesn't (?:this|that|such a thing|anything like this) exist"
                   r"|hasn't anyone|has nobody|has no ?one)\b"
                   r"|\bsurprised (?:there isn't|there is no|there's no|nobody has|no ?one has)\b",
    "nothing_exists": r"\bnothing (?:on the market|out there|like (?:this|that|it) exists)\b|\bdoesn't (?:seem to )?exist yet\b",
    "is_there": rf"\b(?:is|are) there (?:a|an|any)\b{_GAP}{_NOUNS}\b",
    "anyone_know": rf"\bdoes (?:anyone|anybody) (?:know|have|use|recommend) (?:of )?(?:a|an|any)\b{_GAP}{_NOUNS}\b",
    "looking_for": rf"\b(?:looking|searching) for (?:a|an)\b{_GAP}{_NOUNS}\b",
    "alternative_to": r"\b(?:good|decent|cheaper|cheap|free|open[- ]source|better|simpler|lighter) alternatives? (?:to|for)\b",
    "would_pay": r"\b(?:i|we)(?:'d| would| will)? (?:happily |gladly |definitely |totally |easily |absolutely )?pay "
                 r"(?:for|good money|real money|\$|someone|a lot|a subscription)\b|\btake my money\b|\b(?:happily|gladly) pay\b",
    "no_good_way": r"\bthere(?:'s| is| are) no (?:good|decent|simple|easy|reliable|affordable|cheap|sane|proper|great|nice)\b"
                   r"(?:\W+\w+){0,2}?\W+(?:way|tool|app|solution|option|alternative|software|service|product)s?\b"
                   r"|\bno good solution\b",
    "existing_bad": r"\b(?:every|all the) (?:tool|app|solution|product|service)s? (?:i've|i have|we've|we have) (?:tried|used)\b"
                    r"|\bnone of the (?:existing|current|available)\b"
                    r"|\bexisting (?:tools|solutions|apps|products|options|services) (?:are|suck|don't|do not|fall)\b",
    "manual_work": r"\b(?:have|had|need) to (?:\w+ )?(?:manually|by hand)\b|\bwe still manually\b",
    "frustration": r"\b(?:so|really|incredibly|extremely|super|very|insanely) (?:frustrating|annoying|painful|tedious)\b"
                   r"|\b(?:biggest|major|huge) pain(?: point)?\b|\bpain in the (?:ass|butt|neck)\b"
                   r"|\bdrives? me (?:crazy|nuts|insane)\b|\b(?:huge|massive) time sink\b"
                   r"|\bwaste (?:so much|a lot of|hours of) time\b",
    "too_expensive": r"\b(?:way |far |much |ridiculously |absurdly )?(?:too|overly) expensive\b",
}

COMPILED = {name: re.compile(p, re.I) for name, p in PAIN_PATTERNS.items()}

# Excerpts that only vent ("so frustrating", "too expensive") are mostly noise unless an LLM filters them.
WEAK_PATTERNS = {"frustration", "too_expensive"}

# Extra signals for reviews of an existing app or plugin (used only for items of kind "review"): the product
# lacks something, people ask for it, it was taken away, it is paywalled or overpriced, or they switch away.
REVIEW_PATTERNS = {
    "rv_missing": r"\bthere(?:'s| is) no\b|\b(?:it|this app|the app|this plugin|the plugin) (?:has no|doesn't have|does not "
                  r"have|lacks?)\b|\bno (?:way|option|ability|setting) to\b|\black(?:s|ing)? (?:a|an|the|any|of)\b"
                  r"|\b(?:is|are) missing\b|\bmissing (?:a|an|the|feature|features|option|basic|essential)\b"
                  r"|\b(?:only|one) thing (?:missing|it lacks|i miss)\b"
                  r"|\bwould be (?:nice|great|perfect|better|helpful|awesome) if\b"
                  r"|\bi wish (?:it|this|the app|the plugin|they|you) (?:had|would|could|supported|offered|allowed)\b"
                  r"|\b(?:doesn't|does not|won't|can't|cannot) (?:sync|export|import|integrate|connect)\b",
    "rv_request": r"\bplease (?:add|bring back|make it|allow|let us|give us|consider|implement|include|support|offer)\b"
                  r"|\b(?:i'd|i would|would) love (?:to see|it if|an? (?:option|way|feature|setting))\b"
                  r"|\bfeature requests?\b|\bif only (?:it|they|you|the)\b"
                  r"|\b(?:add|need|needs|want) (?:an? )?(?:option|feature|ability|setting|toggle) (?:to|for)\b"
                  r"|\bshould (?:add|allow|support|offer|let us|let you|have an? option)\b",
    "rv_removed": r"\b(?:removed|took away|taken away|got rid of) (?:the|a|an|this|that|support|features?|options?|abilit"
                  r"\w*|capabilit\w*|functionality|access)\b|\b(?:been|was|were|got) (?:removed|taken away|discontinued)\b"
                  r"|\bdiscontinued\b|\bstopped supporting\b"
                  r"|\bno longer (?:has|have|supports?|allows?|lets?|offers?|works? with|available|free)\b"
                  r"|\bused to (?:be able|have|allow|let|be free|offer|support)\b"
                  r"|\b(?:is|are) now (?:only|locked|behind|a premium|paid|part of)\b|\bnow (?:requires?|forces?)\b",
    "rv_price": r"\bpay ?wall(?:ed|s)?\b|\bbehind a pay|\bsubscriptions?\b|\btoo expensive\b|\bover-?priced\b"
                r"|\bprice (?:hike|increase)s?\b|\braised (?:the |their )?prices?\b|\b(?:monthly|yearly|annual) fee\b"
                r"|\blifetime (?:license|purchase|deal|access)\b|\bone[- ]time (?:purchase|payment|fee)\b"
                r"|\bpro version\b|\bpremium (?:version|features?|only|plan|subscription)\b|\b(?:money|cash) grab\b"
                r"|\$\s?\d",
    "rv_switch": r"\b(?:switch(?:ed|ing)?|migrat(?:ed|ing)) (?:back |over )?to\b"
                 r"|\b(?:moved|moving|went back|going back) to (?:another|a different|an? other)\b"
                 r"|\b(?:look(?:ing)? for|find(?:ing)?|found) (?:an?(?:nother| other| different| better)|alternatives?)\b"
                 r"|\bcancel(?:l?ed|l?ing)? (?:my |the )?(?:subscription|plan|membership)\b",
}
REVIEW_COMPILED = {name: re.compile(p, re.I) for name, p in REVIEW_PATTERNS.items()}
# For kind "review": the usual need patterns plus the review ones.
REVIEW_ALL = {**COMPILED, **REVIEW_COMPILED}

# Short trigger phrases only (not the whole match), so the topic words survive neutralize().
_TRIGGERS = re.compile("|".join([
    r"\bi wish (?:there (?:was|were|is)|someone would|somebody would)?",
    r"\bexist(?:s|ed)?(?: yet)?\b",
    r"\b(?:someone|somebody) (?:should|needs? to|ought to|please) (?:build|make|create|write|start)\b",
    r"\bwhy (?:is there no|isn't there|is there not|doesn't|hasn't anyone|has nobody|has no ?one)\b",
    r"\bsurprised (?:there isn't|there is no|there's no|nobody has|no ?one has)\b",
    r"\bnothing (?:on the market|out there)\b",
    r"\b(?:is|are) there (?:a|an|any)\b",
    r"\bdoes (?:anyone|anybody) (?:know|have|use|recommend)(?: of)?\b",
    r"\b(?:i'm |i am |we're |we are )?(?:looking|searching) for\b",
    r"\balternatives? (?:to|for)\b",
    r"\b(?:i|we)(?:'d| would| will)? (?:happily |gladly |definitely |totally |easily |absolutely )?pay(?: for)?\b",
    r"\btake my money\b",
    r"\bthere(?:'s| is| are) no (?:good|decent|simple|easy|reliable|affordable|cheap|sane|proper|great|nice)\b",
    r"\bno good solution\b",
    r"\b(?:every|all the) (?:tool|app|solution|product|service)s? (?:i've|i have|we've|we have) (?:tried|used)\b",
    r"\bnone of the (?:existing|current|available)\b",
    r"\b(?:have|had|need) to manually\b|\bwe still manually\b",
    r"\b(?:so|really|incredibly|extremely|super|very|insanely) (?:frustrating|annoying|painful|tedious)\b",
    r"\b(?:biggest|major|huge) pain(?: point)?\b|\bpain in the (?:ass|butt|neck)\b",
    r"\bdrives? me (?:crazy|nuts|insane)\b|\b(?:huge|massive) time sink\b|\bwaste (?:so much|a lot of|hours of) time\b",
    r"\b(?:way |far |much |ridiculously |absurdly )?(?:too|overly) expensive\b",
]), re.I)

MONEY_PATTERN = re.compile(
    r"\b(?:i|we)(?:'d| would| will)? (?:happily |gladly |definitely |totally |easily |absolutely )?pay\b"
    r"|\btake my money\b|\$\s?\d|€\s?\d|\b\d+(?:\.\d+)?\s?(?:usd|eur|dollars|bucks)\b"
    r"|\b(?:subscription|pricing|per seat|per month|per user|a month)\b|/mo(?:nth)?\b"
    r"|\btoo expensive\b|\b(?:costs?|charges?|charging|charged) (?:us|me)\b|\bwe (?:pay|paid|spend)\b|\bpaying for\b"
    r"|\b(?:i|we) paid (?:for|\$)|\b(?:premium|pro) (?:subscription|version|plan|account|license)\b",
    re.I,
)

_URL = re.compile(r"https?://\S+")


def neutralize(text: str) -> str:
    """Remove the trigger phrases and URLs, so clustering groups by topic, not by phrasing."""
    return re.sub(r"\s+", " ", _URL.sub(" ", _TRIGGERS.sub(" ", text))).strip()


# Extra stop words for keyword extraction (generic words that every cluster shares).
DOMAIN_STOP_WORDS = {
    "wish", "pay", "paid", "paying", "tool", "tools", "app", "apps", "service", "services", "software",
    "looking", "alternative", "alternatives", "exist", "exists", "existed", "someone", "somebody", "build",
    "frustrating", "annoying", "pain", "point", "like", "just", "really", "use", "used", "using", "good",
    "way", "ways", "thing", "things", "know", "think", "people", "don", "doesn", "didn", "isn", "ve", "ll",
    "re", "want", "need", "needs", "make", "time", "would", "could", "lot", "actually", "probably", "sure",
    "yes", "yeah", "maybe", "work", "works", "going", "get", "got", "trying", "tried", "better", "different",
    "able", "does", "did", "say", "said", "kind", "pretty", "lots", "stuff", "hn", "comment", "post",
    "problem", "problems", "solution", "solutions", "product", "products", "existing", "manually",
    "expensive", "money", "month", "year", "years", "day", "days", "new", "great", "nice", "bad", "lack",
    "simple", "easy", "hard", "difficult", "currently", "still", "able", "also", "much", "many", "even",
    # generic review words
    "plugin", "plugins", "feature", "features", "option", "options", "please", "add", "added", "update",
    "updates", "updated", "version", "star", "stars", "review", "reviews", "missing", "removed", "longer",
}

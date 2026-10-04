"""Surface-form augmentation for instructions.

WHY THIS EXISTS. Measured on wiltechs-x-114k (libero_spatial task 7, 20
episodes, identical harness):

    told its own instruction
      "pick up the black bowl on the stove and place it on the plate"        60%
    told ANOTHER TASK's instruction, verbatim from the suite
      "pick up the black bowl on the wooden cabinet and place it on the plate" 0%
    told a PARAPHRASE of its own instruction
      "pick up the black bowl that is on the stove and put it onto the plate"  0%

The middle row looks like obedience -- the video shows the arm going cleanly to
the cabinet. The third row is what it actually means. A model that understood
the sentence would be untouched by `that is` and `put it onto`; this one
collapses. Both rows are explained by the same thing: it has memorised the ~40
instruction strings in the dataset and maps each to a behaviour. Give it a
string from the table and it retrieves; give it anything else and it retrieves
nothing.

The language probe agrees once you read it correctly. It swaps in OTHER TASKS'
instructions, all of which are table entries, so its d(lang) measures "can the
model tell the 40 known strings apart" -- exactly the lookup, and blind to
whether any of it is understanding.

WHAT THIS DOES. Trains on several phrasings per task, sampled per sample per
step, so surface form stops being a usable key. The relation phrase ("on the
stove") is the discriminative content and is varied too, or the model just
memorises relation strings instead of whole ones.

The original string is always in the set: evaluation uses it, and a number
taken on a phrasing the model never saw is not comparable to the ones already
recorded.

Deterministic templates rather than an LLM: the augmentation set has to be
reproducible across runs and machines, or two runs differ by an input
distribution nobody wrote down.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# =========================================================================
# The table. This is what training samples from.
# =========================================================================
# Written out rather than generated, so what the model is trained on can be
# read and reviewed here instead of inferred from template code. Keys are the
# dataset's instruction strings VERBATIM -- a key that differs by a word
# matches nothing and that task trains unaugmented, which is what the trainer
# preflight exists to catch.
#
# Values are ALTERNATES ONLY; the accessor prepends the key. That makes "the
# original is always in the set" structural rather than a rule someone has to
# remember, and eval is run on the original.
#
# Each alternate must preserve: the object, the relation that identifies WHICH
# object, and the destination. The reward is the original task's, so a variant
# that moves any of those trains the policy to do one thing while telling it
# another -- and nothing raises. The variation to aim for is structural
# ("move X to Y", "put X onto Y") and not only lexical, since a model can learn
# a synonym table as easily as it learned the sentences.
#
# All 40 LIBERO instructions, keys verbatim from the suite.
_TABLE: dict[str, list[str]] = {
    "pick up the black bowl between the plate and the ramekin and place it on the plate": [
        "pick up the black bowl that is between the plate and the ramekin and put it on the plate",
        "grasp the black bowl in between the plate and the ramekin and set it on the plate",
        "take the black bowl sitting between the plate and the ramekin and place it onto the plate",
        "lift the black bowl between the plate and the ramekin and put it onto the plate",
        "move the black bowl between the plate and the ramekin to the plate",
        "put the black bowl that sits between the plate and the ramekin onto the plate",
    ],
    "pick up the black bowl next to the ramekin and place it on the plate": [
        "pick up the black bowl that is next to the ramekin and put it on the plate",
        "grasp the black bowl beside the ramekin and set it on the plate",
        "take the black bowl adjacent to the ramekin and place it onto the plate",
        "lift the black bowl next to the ramekin and put it onto the plate",
        "move the black bowl next to the ramekin to the plate",
        "put the black bowl beside the ramekin onto the plate",
    ],
    "pick up the black bowl from table center and place it on the plate": [
        "pick up the black bowl at the center of the table and put it on the plate",
        "grasp the black bowl in the middle of the table and set it on the plate",
        "take the black bowl from the center of the table and place it onto the plate",
        "lift the black bowl from table center and put it onto the plate",
        "move the black bowl at the middle of the table to the plate",
        "put the black bowl from the table center onto the plate",
    ],
    "pick up the black bowl on the cookie box and place it on the plate": [
        "pick up the black bowl that is on the cookie box and put it on the plate",
        "grasp the black bowl on top of the cookie box and set it on the plate",
        "take the black bowl sitting on the cookie box and place it onto the plate",
        "lift the black bowl resting on the cookie box and put it onto the plate",
        "move the black bowl on the cookie box to the plate",
        "put the black bowl that sits on the cookie box onto the plate",
    ],
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": [
        "pick up the black bowl that is in the top drawer of the wooden cabinet and put it on the plate",
        "grasp the black bowl inside the top drawer of the wooden cabinet and set it on the plate",
        "take the black bowl from the top drawer of the wooden cabinet and place it onto the plate",
        "lift the black bowl out of the top drawer of the wooden cabinet and put it onto the plate",
        "move the black bowl in the top drawer of the wooden cabinet to the plate",
        "put the black bowl inside the top drawer of the wooden cabinet onto the plate",
    ],
    "pick up the black bowl on the ramekin and place it on the plate": [
        "pick up the black bowl that is on the ramekin and put it on the plate",
        "grasp the black bowl on top of the ramekin and set it on the plate",
        "take the black bowl sitting on the ramekin and place it onto the plate",
        "lift the black bowl resting on the ramekin and put it onto the plate",
        "move the black bowl on the ramekin to the plate",
        "put the black bowl that sits on the ramekin onto the plate",
    ],
    "pick up the black bowl next to the cookie box and place it on the plate": [
        "pick up the black bowl that is next to the cookie box and put it on the plate",
        "grasp the black bowl beside the cookie box and set it on the plate",
        "take the black bowl adjacent to the cookie box and place it onto the plate",
        "lift the black bowl next to the cookie box and put it onto the plate",
        "move the black bowl next to the cookie box to the plate",
        "put the black bowl beside the cookie box onto the plate",
    ],
    "pick up the black bowl on the stove and place it on the plate": [
        "pick up the black bowl that is on the stove and put it onto the plate",
        "grasp the black bowl on top of the stove and set it on the plate",
        "take the black bowl sitting on the stove and place it onto the plate",
        "lift the black bowl resting on the stove and put it on the plate",
        "move the black bowl on the stove to the plate",
        "put the black bowl that sits on the stove onto the plate",
    ],
    "pick up the black bowl next to the plate and place it on the plate": [
        "pick up the black bowl that is next to the plate and put it on the plate",
        "grasp the black bowl beside the plate and set it on the plate",
        "take the black bowl adjacent to the plate and place it onto the plate",
        "lift the black bowl next to the plate and put it onto the plate",
        "move the black bowl next to the plate onto the plate",
        "put the black bowl beside the plate onto the plate",
    ],
    "pick up the black bowl on the wooden cabinet and place it on the plate": [
        "pick up the black bowl that is on the wooden cabinet and put it on the plate",
        "grasp the black bowl on top of the wooden cabinet and set it on the plate",
        "take the black bowl sitting on the wooden cabinet and place it onto the plate",
        "lift the black bowl resting on the wooden cabinet and put it onto the plate",
        "move the black bowl on the wooden cabinet to the plate",
        "put the black bowl that sits on the wooden cabinet onto the plate",
    ],

    # ---- libero_object ----
    "pick up the orange juice and place it in the basket": [
        "pick up the orange juice and put it in the basket",
        "grasp the orange juice and place it into the basket",
        "take the orange juice and put it into the basket",
        "lift the orange juice and place it inside the basket",
        "move the orange juice into the basket",
        "put the orange juice in the basket",
    ],
    "pick up the ketchup and place it in the basket": [
        "pick up the ketchup and put it in the basket",
        "grasp the ketchup and place it into the basket",
        "take the ketchup and put it into the basket",
        "lift the ketchup and place it inside the basket",
        "move the ketchup into the basket",
        "put the ketchup in the basket",
    ],
    "pick up the cream cheese and place it in the basket": [
        "pick up the cream cheese and put it in the basket",
        "grasp the cream cheese and place it into the basket",
        "take the cream cheese and put it into the basket",
        "lift the cream cheese and place it inside the basket",
        "move the cream cheese into the basket",
        "put the cream cheese in the basket",
    ],
    "pick up the bbq sauce and place it in the basket": [
        "pick up the bbq sauce and put it in the basket",
        "grasp the bbq sauce and place it into the basket",
        "take the bbq sauce and put it into the basket",
        "lift the bbq sauce and place it inside the basket",
        "move the bbq sauce into the basket",
        "put the bbq sauce in the basket",
    ],
    "pick up the alphabet soup and place it in the basket": [
        "pick up the alphabet soup and put it in the basket",
        "grasp the alphabet soup and place it into the basket",
        "take the alphabet soup and put it into the basket",
        "lift the alphabet soup and place it inside the basket",
        "move the alphabet soup into the basket",
        "put the alphabet soup in the basket",
    ],
    "pick up the milk and place it in the basket": [
        "pick up the milk and put it in the basket",
        "grasp the milk and place it into the basket",
        "take the milk and put it into the basket",
        "lift the milk and place it inside the basket",
        "move the milk into the basket",
        "put the milk in the basket",
    ],
    "pick up the salad dressing and place it in the basket": [
        "pick up the salad dressing and put it in the basket",
        "grasp the salad dressing and place it into the basket",
        "take the salad dressing and put it into the basket",
        "lift the salad dressing and place it inside the basket",
        "move the salad dressing into the basket",
        "put the salad dressing in the basket",
    ],
    "pick up the butter and place it in the basket": [
        "pick up the butter and put it in the basket",
        "grasp the butter and place it into the basket",
        "take the butter and put it into the basket",
        "lift the butter and place it inside the basket",
        "move the butter into the basket",
        "put the butter in the basket",
    ],
    "pick up the tomato sauce and place it in the basket": [
        "pick up the tomato sauce and put it in the basket",
        "grasp the tomato sauce and place it into the basket",
        "take the tomato sauce and put it into the basket",
        "lift the tomato sauce and place it inside the basket",
        "move the tomato sauce into the basket",
        "put the tomato sauce in the basket",
    ],
    "pick up the chocolate pudding and place it in the basket": [
        "pick up the chocolate pudding and put it in the basket",
        "grasp the chocolate pudding and place it into the basket",
        "take the chocolate pudding and put it into the basket",
        "lift the chocolate pudding and place it inside the basket",
        "move the chocolate pudding into the basket",
        "put the chocolate pudding in the basket",
    ],

    # ---- libero_goal ----
    "put the bowl on the plate": [
        "place the bowl on the plate",
        "set the bowl on the plate",
        "put the bowl onto the plate",
        "move the bowl to the plate",
        "pick up the bowl and put it on the plate",
        "place the bowl onto the plate",
    ],
    "put the wine bottle on the rack": [
        "place the wine bottle on the rack",
        "set the wine bottle on the rack",
        "put the wine bottle onto the rack",
        "move the wine bottle to the rack",
        "pick up the wine bottle and put it on the rack",
        "place the wine bottle onto the rack",
    ],
    "open the top drawer and put the bowl inside": [
        "open the top drawer and place the bowl inside it",
        "open the top drawer, then put the bowl in it",
        "pull open the top drawer and put the bowl inside",
        "open the top drawer and set the bowl inside",
        "open up the top drawer and place the bowl in it",
        "open the top drawer and put the bowl into it",
    ],
    "put the cream cheese in the bowl": [
        "place the cream cheese in the bowl",
        "put the cream cheese into the bowl",
        "set the cream cheese in the bowl",
        "move the cream cheese into the bowl",
        "pick up the cream cheese and put it in the bowl",
        "place the cream cheese inside the bowl",
    ],
    "put the wine bottle on top of the cabinet": [
        "place the wine bottle on top of the cabinet",
        "set the wine bottle on top of the cabinet",
        "put the wine bottle onto the top of the cabinet",
        "move the wine bottle to the top of the cabinet",
        "pick up the wine bottle and put it on top of the cabinet",
        "stand the wine bottle on top of the cabinet",
    ],
    "push the plate to the front of the stove": [
        "push the plate toward the front of the stove",
        "slide the plate to the front of the stove",
        "push the plate over to the front of the stove",
        "slide the plate toward the front of the stove",
        "push the plate until it is at the front of the stove",
        "push the plate along to the front of the stove",
    ],
    "turn on the stove": [
        "switch on the stove",
        "turn the stove on",
        "switch the stove on",
        "power on the stove",
        "activate the stove",
    ],
    "put the bowl on the stove": [
        "place the bowl on the stove",
        "set the bowl on the stove",
        "put the bowl onto the stove",
        "move the bowl to the stove",
        "pick up the bowl and put it on the stove",
        "place the bowl onto the stove",
    ],
    "put the bowl on top of the cabinet": [
        "place the bowl on top of the cabinet",
        "set the bowl on top of the cabinet",
        "put the bowl onto the top of the cabinet",
        "move the bowl to the top of the cabinet",
        "pick up the bowl and put it on top of the cabinet",
        "place the bowl up on top of the cabinet",
    ],
    "open the middle drawer of the cabinet": [
        "open the cabinet's middle drawer",
        "pull open the middle drawer of the cabinet",
        "open up the middle drawer of the cabinet",
        "slide open the middle drawer of the cabinet",
        "open the middle drawer on the cabinet",
        "pull the middle drawer of the cabinet open",
    ],

    # ---- libero_10 (compositional; the quantifier and the
    # closing action are part of the task, not decoration) ----
    "put the white mug on the left plate and put the yellow and white mug on the right plate": [
        "place the white mug on the left plate and place the yellow and white mug on the right plate",
        "put the white mug onto the left plate, then put the yellow and white mug onto the right plate",
        "set the white mug on the left plate and set the yellow and white mug on the right plate",
        "move the white mug to the left plate and move the yellow and white mug to the right plate",
        "put the white mug on the left plate and the yellow and white mug on the right plate",
        "place the white mug onto the left plate and the yellow and white mug onto the right plate",
    ],
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": [
        "place the white mug on the plate and place the chocolate pudding to the right of the plate",
        "put the white mug onto the plate, then put the chocolate pudding to the right of it",
        "set the white mug on the plate and set the chocolate pudding to the right of the plate",
        "move the white mug to the plate and move the chocolate pudding to the right of the plate",
        "put the white mug on the plate and the chocolate pudding to the right of the plate",
        "place the white mug onto the plate and put the chocolate pudding on the right side of the plate",
    ],
    "put the yellow and white mug in the microwave and close it": [
        "place the yellow and white mug in the microwave and close it",
        "put the yellow and white mug into the microwave, then close it",
        "set the yellow and white mug inside the microwave and shut it",
        "move the yellow and white mug into the microwave and close the door",
        "put the yellow and white mug in the microwave and then close the microwave",
        "place the yellow and white mug inside the microwave and close it",
    ],
    "turn on the stove and put the moka pot on it": [
        "switch on the stove and put the moka pot on it",
        "turn the stove on and place the moka pot on it",
        "turn on the stove, then set the moka pot on it",
        "power on the stove and put the moka pot onto it",
        "switch the stove on and place the moka pot on top of it",
        "turn on the stove and move the moka pot onto it",
    ],
    "put both the alphabet soup and the cream cheese box in the basket": [
        "put both the alphabet soup and the cream cheese box into the basket",
        "place both the alphabet soup and the cream cheese box in the basket",
        "put the alphabet soup and the cream cheese box, both of them, in the basket",
        "move both the alphabet soup and the cream cheese box into the basket",
        "place both of them, the alphabet soup and the cream cheese box, in the basket",
        "put both items in the basket: the alphabet soup and the cream cheese box",
    ],
    "put both the alphabet soup and the tomato sauce in the basket": [
        "put both the alphabet soup and the tomato sauce into the basket",
        "place both the alphabet soup and the tomato sauce in the basket",
        "put the alphabet soup and the tomato sauce, both of them, in the basket",
        "move both the alphabet soup and the tomato sauce into the basket",
        "place both of them, the alphabet soup and the tomato sauce, in the basket",
        "put both items in the basket: the alphabet soup and the tomato sauce",
    ],
    "put both moka pots on the stove": [
        "put both of the moka pots on the stove",
        "place both moka pots on the stove",
        "set both moka pots onto the stove",
        "move both moka pots to the stove",
        "put both the moka pots on the stove",
        "place both of the moka pots onto the stove",
    ],
    "put both the cream cheese box and the butter in the basket": [
        "put both the cream cheese box and the butter into the basket",
        "place both the cream cheese box and the butter in the basket",
        "put the cream cheese box and the butter, both of them, in the basket",
        "move both the cream cheese box and the butter into the basket",
        "place both of them, the cream cheese box and the butter, in the basket",
        "put both items in the basket: the cream cheese box and the butter",
    ],
    "put the black bowl in the bottom drawer of the cabinet and close it": [
        "place the black bowl in the bottom drawer of the cabinet and close it",
        "put the black bowl into the bottom drawer of the cabinet, then close it",
        "set the black bowl inside the bottom drawer of the cabinet and shut it",
        "move the black bowl into the bottom drawer of the cabinet and close the drawer",
        "put the black bowl in the cabinet's bottom drawer and close it",
        "place the black bowl inside the bottom drawer of the cabinet and close it",
    ],
    "pick up the book and place it in the back compartment of the caddy": [
        "pick up the book and put it in the back compartment of the caddy",
        "grasp the book and place it into the back compartment of the caddy",
        "take the book and put it into the back compartment of the caddy",
        "lift the book and place it inside the back compartment of the caddy",
        "move the book into the back compartment of the caddy",
        "put the book in the back compartment of the caddy",
    ],
}


def table_variants(instruction: str) -> list[str] | None:
    """Written variants for `instruction`, original first, or None if absent."""
    key = " ".join(str(instruction).split())
    alts = _TABLE.get(key)
    return None if alts is None else [key] + list(alts)


# =========================================================================
# Template drafting. NOT used during training -- see _sample_paraphrase.
# =========================================================================
# Kept to draft entries for instructions not yet in _TABLE: it emits candidates
# a human then reviews and pastes above. Training reads only _TABLE and
# --paraphrase_file, so a template that mangles a sentence cannot reach the
# model without someone having looked at it.

# Locative relations may take "that is"/"which is"; source/path ones may not
# ("the bowl that is from table center" is not English). Ordered longest-first
# so "next to the" cannot be matched by a shorter prefix.
_LOCATIVE = ("between the", "next to the", "in the top drawer of the",
             "on top of the", "on the", "in the", "inside the", "near the")
_NON_LOCATIVE = ("from the", "from")

_GRASP = ("pick up", "grasp", "take", "lift")
_PLACE_VERBS = ("place", "put", "set")
# Prepositions may only be swapped INSIDE an equivalence class. "place it in
# the basket" -> "place it on the basket" is a different instruction, and the
# reward would still be the original task's -- the model would be trained to
# do one thing while told another. The first version of this file substituted
# on/onto unconditionally and would have done exactly that to libero_object.
_PREP_CLASS = {"on": ("on", "onto"), "onto": ("onto", "on"),
               "in": ("in", "into", "inside"), "into": ("into", "in"),
               "inside": ("inside", "in")}
# "set it into the basket" is not idiomatic; keep `set` on surfaces.
_SET_OK = ("on", "onto")
_QUALIFIER = ("", "that is ", "which is ")

# The relation is OPTIONAL: libero_object is "pick up the alphabet soup and
# place it in the basket", with nothing between object and "and".
_PATTERN = re.compile(
    r"^(?P<verb>pick up|grasp|take|lift)\s+(?P<obj>.+?)"
    r"(?:\s+(?P<rel>(?:%s)\s+.+?|(?:%s)\s+.+?))?\s+"
    r"and\s+(?P<pverb>place|put|set)\s+it\s+"
    r"(?P<prep>onto|on|into|inside|in)\s+(?P<dest>.+)$"
    % ("|".join(re.escape(r) for r in _LOCATIVE),
       "|".join(re.escape(r) for r in _NON_LOCATIVE)),
    re.IGNORECASE)


def paraphrases(instruction: str, limit: int = 8) -> list[str]:
    """Surface variants of one instruction, original first.

    Returns [instruction] alone when the sentence does not match the
    pick-and-place pattern. That is deliberate: a wrong guess at the structure
    would hand the policy a sentence meaning something else, and silently
    train it on the wrong task.
    """
    text = " ".join(str(instruction).split())
    m = _PATTERN.match(text)
    if not m:
        return [text]

    obj, dest = m["obj"], m["dest"]
    rel = m["rel"] or ""
    locative = bool(rel) and rel.lower().startswith(
        tuple(r.lower() for r in _LOCATIVE))
    preps = _PREP_CLASS.get(m["prep"].lower(), (m["prep"].lower(),))

    out, seen = [text], {text.lower()}
    for grasp in _GRASP:
        for qual in (_QUALIFIER if locative else ("",)):
            for pverb in _PLACE_VERBS:
                for prep in preps:
                    if pverb == "set" and prep not in _SET_OK:
                        continue
                    mid = f" {qual}{rel}" if rel else ""
                    cand = f"{grasp} {obj}{mid} and {pverb} it {prep} {dest}"
                    key = cand.lower()
                    if key not in seen:
                        seen.add(key)
                        out.append(cand)
    # Deterministic order, and a stable subset when limit < len(out): the
    # nested loops already vary the grasp verb slowest, so a prefix would be
    # all "pick up". Stride instead, keeping the original at index 0.
    if limit and len(out) > limit:
        rest = out[1:]
        step = len(rest) / (limit - 1)
        out = [out[0]] + [rest[int(i * step)] for i in range(limit - 1)]
    return out


def instruction_strings(raw) -> list[str]:
    """Pull the instruction strings out of whatever `meta.tasks` happens to be.

    lerobot 0.4.0 keeps them in the *index* of a one-column DataFrame whose
    only column is `task_index` (see its `load_tasks`), so the obvious
    `list(tasks)` yields the column NAME -- one bogus instruction called
    "task_index". That is how this was found: the preflight reported
    `1 instructions ... max 1` against a 40-task dataset, and would have gone
    on to reject the run for a coverage gap that did not exist. Older releases
    used a plain {index: task} dict, and a hand-built list is also plausible,
    so all three are accepted.

    Shape alone is ambiguous -- a DataFrame's strings could sit in a column, a
    dict's in either half -- so candidates are ranked by looking like sentences
    rather than trusted positionally.
    """
    if hasattr(raw, "columns"):                               # pandas DataFrame
        cand = [list(raw.index)] + [list(raw[c]) for c in raw.columns]
    elif isinstance(raw, dict):
        cand = [list(raw.values()), list(raw.keys())]
    elif hasattr(raw, "to_list") and hasattr(raw, "index"):   # pandas Series
        cand = [raw.to_list(), list(raw.index)]
    else:
        cand = [list(raw)]
    for c in cand:
        if c and all(isinstance(x, str) for x in c) and any(" " in x for x in c):
            return c
    return [str(x) for x in cand[0]]


def build_table(instructions, limit: int = 8) -> dict[str, list[str]]:
    """instruction -> its variants, for every unique instruction given."""
    table: dict[str, list[str]] = {}
    for ins in instructions:
        key = " ".join(str(ins).split())
        if key not in table:
            table[key] = paraphrases(key, limit)
    return table


def load_table(path: str | Path) -> dict[str, list[str]]:
    """Read a paraphrase table, normalising keys the way lookups will.

    Raises on a variant list that omits its own key: evaluation is run on the
    original string, and a table that dropped it would train a model on
    phrasings it is never scored with.
    """
    data = json.loads(Path(path).read_text())
    table = {}
    for k, v in data.items():
        key = " ".join(str(k).split())
        variants = [" ".join(str(x).split()) for x in v]
        if key not in variants:
            raise ValueError(
                f"paraphrase table entry {key!r} does not contain the original "
                f"string among its variants; eval uses the original, so "
                f"training must see it too")
        table[key] = variants
    return table


def coverage(instructions, limit: int = 8, minimum: int = 5,
             extra: dict[str, list[str]] | None = None):
    """-> (table, under) where `under` lists instructions below `minimum`.

    Templates cover the pick-and-place forms (libero_spatial, libero_object).
    They deliberately do NOT guess at libero_goal's "turn on the stove" or
    libero_10's "put both the alphabet soup and the tomato sauce in the
    basket": inventing structure for those risks emitting a sentence that
    means something else, which trains the wrong task under the right reward.
    Those need a hand-written or LLM-written table, and `under` is the list to
    write it for.

    Partial augmentation is worse than none. Some tasks varied and others not
    means the model can still key on surface form for the unvaried ones, and
    the run tells you nothing about whether augmentation works.
    """
    extra = extra or {}
    table, under = {}, []
    for ins in instructions:
        key = " ".join(str(ins).split())
        if key in table:
            continue
        # Same precedence as training: file, then the written table. Templates
        # are NOT consulted, so this reports what the model would actually see.
        table[key] = extra.get(key) or table_variants(key) or [key]
        if len(table[key]) < minimum:
            under.append(key)
    return table, under


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Report how many variants each instruction would get. Run "
                    "this BEFORE committing a training run to --paraphrase_augment.")
    ap.add_argument("--instructions", help="file with one instruction per line")
    ap.add_argument("--dataset_id", help="read the task strings off a LeRobot dataset")
    ap.add_argument("--extra", help="existing paraphrase JSON to count as covered")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--min_variants", type=int, default=5)
    ap.add_argument("--out", help="write the full table here (JSON)")
    ap.add_argument("--emit", choices=("json", "python", "keys"), default="json",
                    help="keys: just the instruction strings, one per line -- "
                         "the thing to paste when asking someone to write "
                         "entries. python: a _TABLE block ready to paste into "
                         "this file, template drafts where they exist and an "
                         "empty list where they do not.")
    a = ap.parse_args()

    if a.dataset_id:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        # revision="main" for the same reason build_datasets uses it: a dataset
        # without lerobot's codebase version tag otherwise raises
        # RevisionNotFoundError, which current huggingface_hub cannot even
        # construct -- the real cause surfaces as an unrelated TypeError.
        meta = LeRobotDatasetMetadata(a.dataset_id, force_cache_sync=True,
                                      revision="main")
        ins = instruction_strings(meta.tasks)
    elif a.instructions:
        ins = [l for l in Path(a.instructions).read_text().splitlines() if l.strip()]
    else:
        ins = ["pick up the black bowl on the stove and place it on the plate",
               "pick up the alphabet soup and place it in the basket",
               "turn on the stove"]

    table, under = coverage(ins, a.limit, a.min_variants,
                            load_table(a.extra) if a.extra else None)
    for k, v in table.items():
        print(f"{len(v):>3}  {k}")
    print(f"\n{len(table)} instructions, "
          f"{len(table) - len(under)} at >= {a.min_variants} variants, "
          f"{len(under)} BELOW")
    for k in under:
        print(f"  UNDER: {k}")
    if a.emit == "keys":
        print("\n--- instruction strings, verbatim ---")
        for k in table:
            print(k)
    elif a.emit == "python":
        # Drafts come from the TEMPLATES here, unlike the coverage report
        # above: this output is for a human to read and edit before it becomes
        # code, which is the only path by which a template may reach training.
        print("\n--- paste into _TABLE, review every line ---")
        for k in table:
            drafted = [x for x in paraphrases(k, a.limit) if x != k]
            print(f"    {k!r}: [")
            for x in drafted:
                print(f"        {x!r},")
            if not drafted:
                print("        # TEMPLATES DECLINED THIS SENTENCE -- write >= "
                      f"{a.min_variants - 1} by hand.")
                print("        # Keep the object, the thing that identifies "
                      "WHICH object, and the destination.")
            print("    ],")
    if a.out:
        Path(a.out).write_text(json.dumps(table, indent=2, ensure_ascii=False))
        print(f"\nwrote {a.out}")

from __future__ import annotations

import random


TOPICS = [
    "how tides work", "why leaves change color", "the history of paper", "basic cloud formation",
    "how bicycle gears work", "the lifecycle of a honeybee", "why bread rises", "how maps use contour lines",
    "the origins of public libraries", "how acoustic guitars make sound", "why some rocks are magnetic",
    "how eclipses occur", "the basics of composting", "how a mechanical clock keeps time", "the history of umbrellas",
    "how migratory birds navigate", "why soap removes grease", "the formation of river deltas", "how glass is made",
    "the history of postage stamps", "how tree rings form", "the basics of fermentation", "how a camera aperture works",
    "the history of chess", "why the sky changes color at sunset", "how mushrooms reproduce", "the basics of weaving",
    "how glaciers reshape landscapes", "the development of the thermometer", "how popcorn pops", "why metal rusts",
    "the basics of beekeeping", "how suspension bridges carry loads", "the history of pencils", "how dew forms",
    "why cats purr", "the basics of pottery glazing", "how lighthouses evolved", "how seeds germinate",
    "the history of calendars", "how a compass works", "why onions make people cry", "the basics of origami",
    "how coral reefs form", "the history of tea cultivation", "how zippers work", "why snowflakes have six-fold symmetry",
    "how windmills convert energy", "the basics of bookbinding", "how geodes form", "the history of musical notation",
    "how hot-air balloons fly", "why bananas ripen", "the basics of telescope optics", "how canals use locks",
    "the history of ceramics", "how ants organize colonies", "why ice floats", "the basics of knitting",
    "how sundials tell time", "the history of buttons", "how caves form", "why popcorn kernels need moisture",
    "the basics of bonsai", "how barometers measure pressure", "the history of kites", "how dragonflies fly",
    "why salt preserves food", "the basics of mosaics", "how rainbows form", "the history of board games",
    "how water wheels work", "why wool keeps heat in", "the basics of seed saving", "how fossils form",
    "the history of street lighting", "how pulleys provide mechanical advantage", "why apples brown after cutting",
    "the basics of calligraphy", "how stalactites grow", "the history of eyeglasses", "how frogs develop from tadpoles",
    "why copper turns green", "the basics of sourdough", "how hand lenses magnify", "the history of measuring length",
    "how dunes migrate", "why feathers repel some water", "the basics of marbling paper", "how a siphon works",
    "the history of weather vanes", "how moths navigate at night", "why wood grain has patterns", "the basics of terrariums",
    "how simple locks and keys work", "the history of bookmarks", "how frost forms", "why carbonated drinks fizz",
]

TEMPLATES = [
    "Write a self-contained factual passage of about 160 words explaining {topic} to a curious adult. Do not mention these instructions.",
    "Write a clear mini-encyclopedia entry of roughly 160 words about {topic}. Use ordinary prose and no headings.",
    "Explain {topic} in around 160 words for a general reader. Keep it factual, neutral, and self-contained.",
    "Produce a short educational passage, around 160 words, describing {topic}. Avoid lists and do not mention the prompt.",
]


def make_prompt_pool(seed: int = 0) -> dict[str, list[str]]:
    rng = random.Random(seed)
    topics = TOPICS.copy()
    rng.shuffle(topics)
    n = len(topics)
    train_end = int(n * 0.75)
    val_end = int(n * 0.875)
    split_topics = {
        "train": topics[:train_end],
        "val": topics[train_end:val_end],
        "test": topics[val_end:],
    }
    result = {}
    for split, ts in split_topics.items():
        result[split] = [
            TEMPLATES[(i + seed) % len(TEMPLATES)].format(topic=topic)
            for i, topic in enumerate(ts)
        ]
    return result

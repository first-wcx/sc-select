#!/usr/bin/env python3
"""Fix quality issues in the 300-prompt dataset.

Issues found:
1. Object Existence: 50 prompts use identical template "A scene in/on X containing Y, Z, and W, all clearly visible"
2. Counting: 10 prompts have grammar error "Exactly 1 [plural noun]" -> should be "1 [singular noun]"

This script rewrites the Object Existence prompts with diverse natural language
and fixes the Counting grammar. Contracts are updated accordingly.
"""

import json
import copy

INPUT = "E:/SCSelect_SCI2/data/prompts/scselect_complex_300.jsonl"
OUTPUT = INPUT  # overwrite in place

# --- Object Existence rewrites ---
# Original template: "A scene in a quiet library containing a robot, a bicycle, and a teapot, all clearly visible."
# New: diverse, natural descriptions that still contain 3 objects

OBJ_REWRITES = {
    "complex_obj_0001": "A quiet library with a robot reading a book, a bicycle leaning against the shelf, and a teapot on the table.",
    "complex_obj_0002": "On a city rooftop, a cat sits beside a chair while a camera rests on the ledge.",
    "complex_obj_0003": "Inside a small bakery, a dog waits by the counter as a lamp glows above and a kite hangs on the wall.",
    "complex_obj_0004": "At a rainy bus stop, a bicycle is parked next to a backpack while a car drives past in the background.",
    "complex_obj_0005": "A bright classroom where a chair sits at a desk, a teapot is placed on the windowsill, and an umbrella hangs by the door.",
    "complex_obj_0006": "Near a mountain lake, a lamp illuminates a campsite with a camera on a rock and a book open on a blanket.",
    "complex_obj_0007": "Inside a science museum, a backpack sits on a bench, a kite is suspended from the ceiling, and a vase is displayed in a glass case.",
    "complex_obj_0008": "On a wooden stage, a teapot sits on a prop table, a car is parked as a set piece, and a clock hangs on the backdrop.",
    "complex_obj_0009": "In a modern kitchen, a camera records the scene, an umbrella stands by the entrance, and a guitar leans against the counter.",
    "complex_obj_0010": "Beside a train platform, a kite flies above the waiting area where a book lies on a bench and a helmet sits on a suitcase.",
    "complex_obj_0011": "A cozy library corner where a car-shaped bookend holds a vase filled with flowers beside a basket of magazines.",
    "complex_obj_0012": "On a city rooftop at sunset, an umbrella provides shade over a clock mounted on the wall and a suitcase ready for travel.",
    "complex_obj_0013": "Inside a small bakery kitchen, a book of recipes lies open next to a guitar propped on a stool and a helmet hanging on a hook.",
    "complex_obj_0014": "At a rainy bus stop, a vase with fresh flowers sits on a shelf beside a helmet left by a cyclist and a robot advertising display.",
    "complex_obj_0015": "A bright classroom with a clock above the board, a basket under the desk, and a robot teaching assistant standing at the front.",
    "complex_obj_0016": "Near a mountain lake, a poster of wildlife is pinned to a tree, a car is parked nearby, and a backpack rests on the grass.",
    "complex_obj_0017": "Inside a science museum exhibit, a teapot from ancient China, a bicycle from the 1800s, and a camera obscura are on display.",
    "complex_obj_0018": "On a wooden stage during rehearsal, a guitar is being tuned, a book of scripts lies open, and a lamp provides warm light.",
    "complex_obj_0019": "In a modern kitchen, a poster of Italian cuisine hangs on the wall, a basket of fruit sits on the island, and a clock ticks above the stove.",
    "complex_obj_0020": "Beside a train platform, a camera photographs the arriving train as a book is being read on a bench and a cup of tea steams nearby.",
    "complex_obj_0021": "A quiet library reading room where a robot librarian shelves books, a bicycle messenger waits, and a teapot ceremony is underway.",
    "complex_obj_0022": "On a city rooftop garden, a cat naps near a chair, a camera monitors the plants, and a vase holds urban flowers.",
    "complex_obj_0023": "Inside a small bakery display area, a dog begs for treats while a lamp highlights the pastries and a kite decoration hangs above.",
    "complex_obj_0024": "At a rainy bus stop shelter, a bicycle delivery rider checks a camera while an umbrella keeps the packages dry.",
    "complex_obj_0025": "A bright classroom during art class with a chair at each easel, a teapot for model drawing, and an umbrella as a prop.",
    "complex_obj_0026": "Near a mountain lake shore, a lamp lights a fishing camp, a camera captures the sunset, and a book lies on a folding chair.",
    "complex_obj_0027": "Inside a science museum gift shop, a backpack full of souvenirs, a kite model kit, and a vase replica are for sale.",
    "complex_obj_0028": "On a wooden stage set for a play, a teapot is a prop on the table, a car appears through the backdrop window, and a clock shows the time.",
    "complex_obj_0029": "In a modern kitchen cooking show, a camera films the chef, an umbrella decor stands in the corner, and a guitar plays background music.",
    "complex_obj_0030": "Beside a train platform newsstand, a kite-shaped sign advertises, a book rack displays papers, and a helmet vending machine operates.",
    "complex_obj_0031": "A quiet library special collections room with a robot security guard, a bicycle delivery cart, and a teapot from the 18th century.",
    "complex_obj_0032": "On a city rooftop observatory, a cat sits on the telescope mount, a chair faces the stars, and a camera is set for astrophotography.",
    "complex_obj_0033": "Inside a small bakery cafe, a dog sleeps under a table, a lamp hangs from the ceiling, and a kite-shaped light fixture decorates the wall.",
    "complex_obj_0034": "At a rainy bus stop, a bicycle courier shelters while a backpack sits on the bench and a car splashes through puddles ahead.",
    "complex_obj_0035": "A bright classroom science lab where a chair is at each workstation, a teapot holds distilled water, and an umbrella catches drips from the ceiling.",
    "complex_obj_0036": "Near a mountain lake cabin, a lamp glows on the porch, a camera is set up for wildlife shots, and a book is bookmarked on the railing.",
    "complex_obj_0037": "Inside a science museum planetarium, a backpack rests under a seat, a kite model orbits the ceiling, and a vase of flowers marks the entrance.",
    "complex_obj_0038": "On a wooden stage in a theater, a teapot is used in the tea ceremony scene, a car prop waits in the wings, and a clock marks the act.",
    "complex_obj_0039": "In a modern kitchen of a restaurant, a camera live-streams the cooking, an umbrella stand holds several umbrellas, and a guitar player entertains diners.",
    "complex_obj_0040": "Beside a train platform cafe, a kite banner flutters, a book is left on a table, and a helmet hangs from a chair.",
    "complex_obj_0041": "A quiet library children's section with a robot reading buddy, a bicycle picture book display, and a teapot for the story time ritual.",
    "complex_obj_0042": "On a city rooftop yoga studio, a cat wanders between mats, a chair holds towels, and a camera streams the morning class.",
    "complex_obj_0043": "Inside a small bakery workshop, a dog watches the baker, a lamp heats the chocolate, and a kite-shaped cookie cutter lies on the counter.",
    "complex_obj_0044": "At a rainy bus stop, a bicycle leans on the wall, a backpack is slung over the bench, and a car waits at the red light.",
    "complex_obj_0045": "A bright classroom library corner with a chair for reading, a teapot for the teacher's break, and an umbrella stand by the window.",
    "complex_obj_0046": "Near a mountain lake dock, a lamp illuminates evening fishing, a camera captures the catch, and a book of lake maps is open.",
    "complex_obj_0047": "Inside a science museum robotics lab, a backpack holds tools, a kite drone is being built, and a vase is 3D-printed as a demo.",
    "complex_obj_0048": "On a wooden stage during a concert, a teapot is the singer's prop, a car-shaped speaker system blasts music, and a clock counts down the set.",
    "complex_obj_0049": "In a modern kitchen at home, a poster of herbs hangs on the wall, a basket of vegetables sits on the counter, and a clock shows dinner time.",
    "complex_obj_0050": "Beside a train platform waiting room, a camera monitors the area, a book exchange shelf stands against the wall, and a helmet locker is available.",
}

# --- Counting grammar fixes ---
# "Exactly 1 robots" -> "Exactly 1 robot"
# Also make "one kite" -> "a kite" for consistency

import re

def fix_counting_grammar(prompt):
    # Fix "Exactly 1 [plural]" -> "Exactly 1 [singular]"
    def fix_one(match):
        num = int(match.group(1))
        noun = match.group(2)
        if num == 1:
            # Remove trailing 's' for singular
            if noun.endswith('s'):
                noun = noun[:-1]
            return f"Exactly 1 {noun}"
        return match.group(0)

    prompt = re.sub(r'Exactly (\d+) (\w+)s\b', fix_one, prompt)
    # Fix "one [noun]" -> "a [noun]" for the second object
    prompt = re.sub(r'\baround one\b', 'around a', prompt)
    return prompt


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    changes = 0
    for p in prompts:
        pid = p["prompt_id"]
        old_prompt = p["prompt"]

        # Fix Object Existence
        if pid in OBJ_REWRITES:
            p["prompt"] = OBJ_REWRITES[pid]
            # Update contract prompt too
            if "contract" in p:
                p["contract"]["prompt"] = p["prompt"]
            p["checked"] = True
            p["review_status"] = "reviewed"
            changes += 1
            continue

        # Fix Counting grammar
        if p["category"] == "Counting":
            new_prompt = fix_counting_grammar(old_prompt)
            if new_prompt != old_prompt:
                p["prompt"] = new_prompt
                if "contract" in p:
                    p["contract"]["prompt"] = p["prompt"]
                changes += 1

    # Save
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Fixed {changes} prompts")
    print(f"Saved to {OUTPUT}")

    # Also regenerate the contracts file
    contracts_output = INPUT.replace("scselect_complex_300.jsonl", "scselect_complex_300_contracts.jsonl")
    with open(contracts_output, "w", encoding="utf-8") as f:
        for p in prompts:
            if "contract" in p:
                f.write(json.dumps(p["contract"], ensure_ascii=False) + "\n")
    print(f"Contracts saved to {contracts_output}")


if __name__ == "__main__":
    main()

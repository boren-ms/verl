"""Compare openasr_en vs english text normalizers."""

from recipe.phimm.utils.tn import TN_DICT

en = TN_DICT["english"]
openasr = TN_DICT["openasr_en"]

test_cases = [
    "Hello, World!!!",
    "It's a test.",
    "Mr. Smith went to Washington D.C.",
    "I have $100 and 50%.",
    "The U.S.A. is a country.",
    "He said, \"Don't go!\"",
    "3.14 is approximately pi.",
    "I can't believe it's not butter.",
    "Dr. Jones & Mr. Hyde",
    "1st, 2nd, 3rd, 4th place",
    "COVID-19 was in 2020.",
    "She'll be comin' 'round the mountain.",
    "The café served crème brûlée.",
    "10:30 AM on 01/15/2024",
    "won't shouldn't couldn't wouldn't",
    "   extra   spaces   everywhere   ",
    "UPPERCASE AND lowercase MiXeD",
    "email@example.com is an address",
    "100kg weighs 220lbs",
    # British/American spelling variants (normalizer divergence candidates)
    "The archaeology site was impressive.",
    "The cancellation was unexpected.",
    "The cancelation was unexpected.",
    "He was pummelling the bag.",
    "The snowploughs cleared the road.",
    "She wrote a travelogue about Europe.",
    "",
]

print(f"{'Input':<50} | {'english':<50} | {'openasr_en':<50} | {'Match'}")
print("-" * 160)

diffs = 0
for text in test_cases:
    r_en = en(text)
    r_oa = openasr(text)
    match = "✓" if r_en == r_oa else "✗"
    if r_en != r_oa:
        diffs += 1
    print(f"{text:<50} | {r_en:<50} | {r_oa:<50} | {match}")

print(f"\n{diffs} differences out of {len(test_cases)} test cases")

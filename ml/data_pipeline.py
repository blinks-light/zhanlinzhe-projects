"""Generate the deterministic synthetic email dataset used for training.

The dataset is intentionally small (toy scale): it exists to demonstrate the
data versioning workflow, not to build a production-grade classifier.

Usage:
    python ml/data_pipeline.py --version v1 --out data/emails.csv
    python ml/data_pipeline.py --version v2 --out data/emails.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDNAMES = ["text", "is_spam", "category"]

SPAM_EMAILS = [
    "Congratulations! You have won a $1000 gift card. Click http://claim-prize.example to claim it.",
    "URGENT: your bank account will be suspended. Verify your identity at http://secure.example",
    "Buy cheap replica watches at 90% discount. Limited time offer, order today!",
    "You are pre-approved for a $50,000 loan with zero interest. Reply YES to get cash today.",
    "Your PayPal account has been limited. Confirm your password here: http://paypa1.example",
    "Lose 10 kg in 7 days with this miracle pill. Order now and get a free bottle!",
    "Dear customer, your parcel is held at customs. Pay the clearance fee at http://fee.example",
    "Invest $250 and earn $5000 in one week with our crypto trading bot. Guaranteed profit!",
    "Final notice: your subscription payment failed. Update your card at http://billing.example",
    "You have a pending refund of $899. Send your bank details to claim it.",
    "Meet singles in your area tonight! Click the link to see profiles.",
    "Get rich fast with this secret investment plan. Limited spots available, act now!",
    "Your iPhone 17 lottery ticket won! Provide your address to receive the prize.",
    "Exclusive offer: 100% free casino bonus, play and win big now.",
    "Verify your account within 24 hours or it will be permanently deleted. Click here.",
    "Cheap medications online, no prescription needed. Discreet shipping worldwide.",
    "Work from home and earn $3000 per week. No experience required, sign up today!",
    "Your Netflix payment was declined. Update your billing information immediately.",
    "Congratulations, you are our lucky winner! Reply with your credit card number.",
    "URGENT: your computer is infected with a virus. Call this number for support.",
    "Claim your free trial of the miracle skincare cream before the offer expires.",
    "You have been selected for a government grant of $25,000. Apply now, no fees.",
    "Hot stock tip: buy NOW before the price explodes. Guaranteed 500% returns.",
    "Your account has unusual activity. Confirm your login at http://account-check.example",
    "Lowest mortgage rates ever! Refinance today and save thousands.",
    "Free entry to the international lottery, just send us your ID and bank details.",
    "Act now: 80% off all designer bags, today only. Shop the secret sale.",
    "Dear winner, your email was selected in our annual draw. Claim your cash prize.",
    "Increase your followers by 10,000 in one day. Buy cheap likes and views now.",
    "Your delivery could not be completed. Pay $2.99 redelivery fee at http://track.example",
]

WORK_EMAILS = [
    "Could you review the Q3 budget spreadsheet before Friday's meeting?",
    "Please send the updated project timeline to the client by end of day.",
    "The sprint planning meeting is moved to 10 AM in room B.",
    "I pushed the fix for the login bug, please review the pull request.",
    "Here are the meeting notes and action items from this morning.",
    "Can we schedule a call to discuss the vendor contract renewal?",
    "The quarterly report draft is ready for your comments.",
    "Please approve the purchase order for the new laptops.",
    "Our team standup will start fifteen minutes later tomorrow.",
    "Attached is the revised proposal for the marketing campaign.",
    "We need to finalize the hiring plan before the board meeting.",
    "The client asked for a demo of the new dashboard next Tuesday.",
    "Please update the risk register with the latest findings.",
    "I will be on leave next week, please contact Anna for urgent issues.",
    "The deployment window is confirmed for Thursday at 8 PM.",
    "Let's review the API design document in the afternoon.",
    "Could you share the sales numbers for last month?",
    "The audit checklist needs your signature by Friday.",
    "We should align on the roadmap priorities for next quarter.",
    "Please book the conference room for the interview panel.",
    "The invoice from the supplier is missing the PO number.",
    "Great work on the release, the customer feedback is positive.",
    "Can you prepare the slides for the steering committee?",
    "The security review found two medium issues, see the ticket.",
]

PERSONAL_EMAILS = [
    "Let's grab coffee this weekend and catch up, it has been a while.",
    "Happy birthday! Hope you have a wonderful day with your family.",
    "We are planning a barbecue on Sunday, want to come over?",
    "Thanks for dinner last night, we should do it again soon.",
    "Mom asked if you are coming home for the holidays.",
    "I booked the tickets for our trip to the mountains next month.",
    "Can you send me the photos from the wedding?",
    "Are you free for lunch on Wednesday? There is a new place downtown.",
    "The kids had a great time at the park today.",
    "I finally finished that book you recommended, loved the ending.",
    "Let's go for a run tomorrow morning before work.",
    "Do you want to watch the game at my place on Saturday?",
    "Grandma's birthday party is at 6 PM, don't forget the cake.",
    "I adopted a puppy last week, sending pictures soon.",
    "Thanks for helping me move, I owe you a beer.",
    "Our anniversary dinner is booked for Friday at 8.",
    "How was your vacation? Tell me everything.",
    "I'm thinking of taking a cooking class, want to join?",
    "The concert was amazing, you should have come.",
    "Can I borrow your ladder this weekend?",
    "We missed you at the reunion, everyone asked about you.",
    "Movie night at ours on Friday, bring snacks.",
]

NOTIFICATION_EMAILS = [
    "Your package has been delivered to the front desk at 3 PM.",
    "Reminder: your dentist appointment is scheduled for Monday at 9 AM.",
    "Your order #A1023 has shipped and will arrive on Thursday.",
    "Your password was changed successfully. If this was not you, contact support.",
    "Your monthly invoice is now available in your account.",
    "Your subscription will renew automatically on November 1.",
    "A new login to your account was detected from a Windows device.",
    "Your flight check-in is now open, boarding starts at 6:40 AM.",
    "Your appointment with Dr. Lee has been confirmed for Tuesday.",
    "Your food delivery is on the way and will arrive in 12 minutes.",
    "Your hotel booking confirmation for December 12 is attached.",
    "Your bank statement for September is ready to download.",
    "Your verification code is 483920, it expires in 10 minutes.",
    "Your car service is booked for Saturday at 11 AM.",
    "Your library book is due on October 30.",
    "Your gym membership renews next week.",
    "Your train ticket for the 8:15 service is confirmed.",
    "Your report card is available in the student portal.",
    "Your electricity bill of $74.20 is due on October 25.",
    "Your email storage is almost full, please free up space.",
    "Your ride is arriving in 3 minutes at the main entrance.",
    "Your seat for the conference has been confirmed.",
    "Your parcel is at the local pickup point, bring your ID.",
    "Your account statement is ready, log in to view it.",
]

# v2 removes four low-information / near-duplicate spam messages.
REMOVED_IN_V2 = {
    "Your iPhone 17 lottery ticket won! Provide your address to receive the prize.",
    "Exclusive offer: 100% free casino bonus, play and win big now.",
    "You have been selected for a government grant of $25,000. Apply now, no fees.",
    "Free entry to the international lottery, just send us your ID and bank details.",
}

# v1 mislabels these two notifications as work emails; v2 fixes the labels.
MISLABELED_IN_V1 = {
    "Your monthly invoice is now available in your account.",
    "Your account statement is ready, log in to view it.",
}

# v2 adds hard negatives: legitimate emails that contain promotional wording.
HARD_NEGATIVES = [
    {
        "text": "The vendor offered a free trial of the analytics tool, let's evaluate it.",
        "is_spam": 0,
        "category": "work",
    },
    {
        "text": "Reminder: the deadline for the budget report is Friday, submit your numbers.",
        "is_spam": 0,
        "category": "work",
    },
    {
        "text": "Your discount code for the company store expires tomorrow.",
        "is_spam": 0,
        "category": "notification",
    },
    {
        "text": "Congratulations to the team on winning the client proposal!",
        "is_spam": 0,
        "category": "work",
    },
    {
        "text": "Your appointment for the annual health check is confirmed for Monday.",
        "is_spam": 0,
        "category": "notification",
    },
    {
        "text": "We won the contract! Celebration lunch on Friday at noon.",
        "is_spam": 0,
        "category": "work",
    },
]


def _base_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for text in SPAM_EMAILS:
        rows.append({"text": text, "is_spam": 1, "category": "spam"})
    for text in WORK_EMAILS:
        rows.append({"text": text, "is_spam": 0, "category": "work"})
    for text in PERSONAL_EMAILS:
        rows.append({"text": text, "is_spam": 0, "category": "personal"})
    for text in NOTIFICATION_EMAILS:
        rows.append({"text": text, "is_spam": 0, "category": "notification"})
    return rows


def build_v1() -> list[dict[str, object]]:
    """First usable dataset: 100 rows, with two known label mistakes."""
    rows = _base_rows()
    for row in rows:
        if row["text"] in MISLABELED_IN_V1:
            row["category"] = "work"
    return rows


def build_v2() -> list[dict[str, object]]:
    """Cleaned dataset: fixed labels, no duplicates, six hard negatives."""
    rows = [row for row in _base_rows() if row["text"] not in REMOVED_IN_V2]
    rows.extend(dict(row) for row in HARD_NEGATIVES)
    return rows


def validate(rows: list[dict[str, object]]) -> None:
    if len(rows) != len({row["text"] for row in rows}):
        raise ValueError("duplicate email texts found")
    for row in rows:
        is_spam = int(row["is_spam"])
        category = str(row["category"])
        if is_spam not in (0, 1):
            raise ValueError(f"invalid is_spam value: {is_spam}")
        if (category == "spam") != (is_spam == 1):
            raise ValueError(f"label mismatch for: {row['text']}")
        if not str(row["text"]).strip():
            raise ValueError("empty email text")


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> str:
    counts = {
        category: sum(1 for row in rows if row["category"] == category)
        for category in ("spam", "work", "personal", "notification")
    }
    spam_count = sum(int(row["is_spam"]) for row in rows)
    details = ", ".join(f"{name}={count}" for name, count in counts.items())
    return f"rows={len(rows)} spam={spam_count} ({details})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--out", default="data/emails.csv")
    args = parser.parse_args()

    rows = build_v1() if args.version == "v1" else build_v2()
    validate(rows)
    write_csv(rows, Path(args.out))
    print(f"wrote dataset {args.version} to {args.out}: {summarize(rows)}")


if __name__ == "__main__":
    main()

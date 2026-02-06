from src.pmeru.data.event_stream import EventStreamDataset, SyntheticEventConfig
import logging

# Mute logger
logging.basicConfig(level=logging.ERROR)


def test_rules():
    config = SyntheticEventConfig(num_events=2000)

    # Mock tokenizer
    class MockTok:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1]}

        def decode(self, *args):
            return ""

    ds = EventStreamDataset(MockTok(), config)

    violations = 0
    denials = 0
    high_risks = 0

    print("Checking 2000 events for compliance rules...")
    for i in range(2000):
        evt = ds._generate_event(i)

        if evt.risk_score >= 80:
            high_risks += 1
            if evt.status == "DENIED":
                denials += 1
            else:
                print(f"VIOLATION: Risk {evt.risk_score} -> Status {evt.status}")
                violations += 1

    print(f"Total High Risk (>80): {high_risks}")
    print(f"Total Denied: {denials}")
    print(f"Violations: {violations}")

    if violations == 0 and high_risks > 0:
        print("PASS: Rigid Compliance Logic verified.")
    elif high_risks == 0:
        print("WARN: No high risk events generated to test.")
    else:
        print("FAIL: Rule violations found.")


if __name__ == "__main__":
    test_rules()

import tracemalloc
from dotenv import load_dotenv
from src.util import setup_logging
from src.graph import run_debate
from src.report import generate_report

logger = setup_logging('main')

def main():
    load_dotenv()
    tracemalloc.start()

    debate_result = run_debate('AAPL', mode='debate')
    report = generate_report(debate_result)

    # Print summary
    consensus = report['consensus']
    print(f"\n{'='*60}")
    print(f"  {report['ticker']} Stock Analysis Report")
    print(f"  Generated: {report['timestamp']}")
    print(f"{'='*60}")
    print(f"  Current Price: ${report['current_price']:.2f}")
    print(f"  Consensus:     {consensus['recommendation']}")
    print(f"  Conviction:    {consensus['avg_conviction']:.1f}/10")
    print(f"  Agreement:     {consensus['agreement']}/{consensus['total_agents']} agents")
    print(f"{'='*60}\n")

    for agent_name, agent_data in report['agents'].items():
        print(f"--- {agent_data['display_name']} ---")
        print(f"  Rating: {agent_data['recommendation']} | Conviction: {agent_data['conviction']:.0f}/10")
        if agent_data['risks']:
            print(f"  Risks: {agent_data['risks'][:200]}")
        print()


if __name__ == '__main__':
    main()

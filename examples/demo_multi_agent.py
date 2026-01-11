"""
Demo script to test Multi-Agent System.
Run this to verify the framework works correctly.
"""
import sys
import os
import time

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.orchestrator import AgentOrchestrator
from src.agents.demo_agents import CounterAgent, FileProcessorAgent

def main():
    print("=" * 60)
    print("Multi-Agent System Demo")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Add multiple agents
    orchestrator.add_agent(CounterAgent("Counter-1", max_count=5))
    orchestrator.add_agent(CounterAgent("Counter-2", max_count=8))
    orchestrator.add_agent(FileProcessorAgent(
        "FileProc-1", 
        files=["file1.pdf", "file2.docx", "file3.txt"]
    ))
    
    print(f"\n✓ Created {len(orchestrator.agents)} agents")
    
    # Start all agents in parallel
    print("\n🚀 Starting all agents...\n")
    orchestrator.start_all()
    
    # Monitor progress
    while not orchestrator.is_complete():
        progress = orchestrator.get_progress()
        print(f"Overall Progress: {progress['overall_progress']*100:.1f}% | "
              f"Running: {progress['running']} | "
              f"Completed: {progress['completed']} | "
              f"Failed: {progress['failed']}")
        
        # Show individual agent progress
        for agent_status in progress['agents']:
            status_icon = {
                'idle': '⏸️',
                'running': '🏃',
                'completed': '✅',
                'failed': '❌'
            }.get(agent_status['status'], '❓')
            
            print(f"  {status_icon} {agent_status['name']}: "
                  f"{agent_status['progress']*100:.1f}% "
                  f"[{agent_status['status']}]")
        
        print()  # Blank line
        time.sleep(1)
    
    # Get final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    results = orchestrator.get_results()
    for agent_name, data in results.items():
        print(f"\n{agent_name}:")
        print(f"  Status: {data['status']}")
        if data['error']:
            print(f"  Error: {data['error']}")
        else:
            print(f"  Result: {data['result']}")
    
    print("\n✅ All agents completed!")

if __name__ == "__main__":
    main()

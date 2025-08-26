

import asyncio
from agents.langgrap_context_system import LangGraphContextSystem


async def main():
  """Main function to demonstrate the system"""
  print("🚀 Initializing LangGraph Advanced Context Management System...")
  
  # Initialize the system
  system = LangGraphContextSystem()
  
  # Test scenarios
  test_requests = [
    "Analyze the code structure of my Python project and suggest improvements",
    "Help me refactor a large function into smaller, more maintainable pieces",
    "Review my code for performance bottlenecks and optimization opportunities",
    "Generate comprehensive documentation for my API endpoints",
    "Create unit tests for my new authentication module"
  ]
  
  print("\n📊 Processing test requests...")
  
  for i, request in enumerate(test_requests, 1):
    print(f"\n--- Test {i}: {request[:50]}... ---")
    
    result = await system.process_request(request)
    
    if result['success']:
      print(f"✅ Success: {result['response']}")
      print(f"📝 Context tokens used: {result['context_used']}")
      print(f"🧠 Memory: {result['memory_summary']}")
      print(f"📈 Metrics: {result['performance_metrics']}")
    else:
      print(f"❌ Error: {result['error']}")
    
    # Simulate some delay
    await asyncio.sleep(0.5)
  
  print("\n🎯 System Performance Summary:")
  print(f"Total interactions: {system.root_agent.performance_metrics['interactions']}")
  print(f"Context retrievals: {system.root_agent.performance_metrics['context_retrievals']}")
  print(f"Memory items - Working: {len(system.memory_hierarchy.working_memory.items)}")
  print(f"Memory items - Short-term: {len(system.memory_hierarchy.short_term_memory.items)}")
  print(f"Memory items - Long-term: {len(system.memory_hierarchy.long_term_memory.items)}")
  print(f"Episodic memories: {len(system.memory_hierarchy.episodic_memory)}")

if __name__ == "__main__":
  print("LangGraph Advanced Context Management System")
  print("=" * 50)
  asyncio.run(main())
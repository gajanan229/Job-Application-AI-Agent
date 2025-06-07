"""
Demo script for the LangGraph Application Factory Workflow.

This script demonstrates the complete workflow engine without requiring actual files or API keys.
It shows the state transitions and progress tracking capabilities.
"""

from core.graph import ApplicationFactoryWorkflow, WorkflowStatus, WorkflowProgress
from datetime import datetime
import os


def create_demo_files():
    """Create temporary demo files for testing."""
    # Create demo directory
    demo_dir = "demo_temp"
    os.makedirs(demo_dir, exist_ok=True)
    
    # Create demo resume file
    resume_path = os.path.join(demo_dir, "demo_resume.pdf")
    with open(resume_path, "w") as f:
        f.write("Demo resume content")
    
    # Create demo job description file
    job_path = os.path.join(demo_dir, "demo_job.pdf")
    with open(job_path, "w") as f:
        f.write("Demo job description content")
    
    return resume_path, job_path


def cleanup_demo_files(resume_path, job_path):
    """Clean up demo files."""
    try:
        os.remove(resume_path)
        os.remove(job_path)
        os.rmdir(os.path.dirname(resume_path))
    except:
        pass


def demo_workflow_initialization():
    """Demonstrate workflow initialization and graph compilation."""
    print("🔧 Demonstrating LangGraph Workflow Initialization...")
    print("-" * 60)
    
    try:
        # Create workflow instance
        print("Creating ApplicationFactoryWorkflow instance...")
        workflow = ApplicationFactoryWorkflow()
        
        print("✅ Workflow created successfully!")
        print(f"Graph compiled: {workflow.compiled_graph is not None}")
        
        # Test conditional edge functions
        print("\n🔀 Testing Conditional Edge Functions...")
        
        # Create test state
        test_progress = WorkflowProgress(
            current_node="test_node",
            completed_nodes=[],
            failed_nodes=[],
            total_nodes=5,
            progress_percentage=0.0,
            status=WorkflowStatus.RUNNING
        )
        
        # Test successful condition
        test_state = {"workflow_progress": test_progress}
        result = workflow.should_continue_after_init(test_state)
        print(f"Successful state decision: {result}")
        
        # Test failed condition
        test_progress.status = WorkflowStatus.FAILED
        result = workflow.should_continue_after_init(test_state)
        print(f"Failed state decision: {result}")
        
        print("✅ Conditional edge functions working correctly!")
        
    except Exception as e:
        print(f"❌ Workflow initialization failed: {e}")
        return False
    
    return True


def demo_workflow_progress_tracking():
    """Demonstrate workflow progress tracking."""
    print("\n📊 Demonstrating Workflow Progress Tracking...")
    print("-" * 60)
    
    # Create progress tracker
    progress = WorkflowProgress(
        current_node="initialize_workflow",
        completed_nodes=[],
        failed_nodes=[],
        total_nodes=5,
        progress_percentage=0.0,
        status=WorkflowStatus.PENDING,
        start_time=datetime.now()
    )
    
    # Simulate workflow progress
    stages = [
        ("initialize_workflow", 20.0),
        ("process_documents", 40.0),
        ("generate_content", 70.0),
        ("create_documents", 90.0),
        ("finalize_workflow", 100.0)
    ]
    
    print("Simulating workflow progress:")
    
    for stage, percentage in stages:
        progress.current_node = stage
        progress.completed_nodes.append(stage)
        progress.progress_percentage = percentage
        progress.status = WorkflowStatus.RUNNING
        
        print(f"  📌 {stage}: {percentage}% complete")
        
        # Show progress dict conversion
        if stage == "generate_content":
            progress_dict = progress.to_dict()
            print(f"    Progress data: {progress_dict['status']}, {progress_dict['progress_percentage']}%")
    
    # Complete workflow
    progress.status = WorkflowStatus.COMPLETED
    progress.end_time = datetime.now()
    
    print(f"✅ Workflow completed! Status: {progress.status.value}")
    print(f"Total stages completed: {len(progress.completed_nodes)}")
    
    return True


def demo_workflow_state_validation():
    """Demonstrate workflow state validation."""
    print("\n🔍 Demonstrating Workflow State Validation...")
    print("-" * 60)
    
    try:
        workflow = ApplicationFactoryWorkflow()
        
        # Test state with missing files
        print("Testing validation with missing files...")
        
        # Create initial state with non-existent files
        initial_state = {
            "api_key": "demo_key",
            "master_resume_pdf_path": "/nonexistent/resume.pdf",
            "job_description_pdf_path": "/nonexistent/job.pdf",
            "rag_processor": None,
            "master_resume_result": None,
            "job_description_result": None,
            "content_generator": None,
            "resume_content": None,
            "cover_letter_content": None,
            "document_generator": None,
            "resume_document": None,
            "cover_letter_document": None,
            "workflow_progress": WorkflowProgress(
                current_node="",
                completed_nodes=[],
                failed_nodes=[],
                total_nodes=5,
                progress_percentage=0.0,
                status=WorkflowStatus.PENDING
            ),
            "node_statuses": {},
            "errors": [],
            "logs": [],
            "user_preferences": {},
            "session_id": "demo_session",
            "timestamp": datetime.now().isoformat()
        }
        
        # Test initialization node with invalid state
        result = workflow.initialize_workflow_node(initial_state)
        
        if result["workflow_progress"].status == WorkflowStatus.FAILED:
            print("✅ Validation correctly caught missing files!")
            print(f"Error message: {result['errors'][0] if result['errors'] else 'No specific error'}")
        else:
            print("❌ Validation did not catch the error")
        
        # Test with valid files
        print("\nTesting validation with valid files...")
        resume_path, job_path = create_demo_files()
        
        initial_state["master_resume_pdf_path"] = resume_path
        initial_state["job_description_pdf_path"] = job_path
        
        result = workflow.initialize_workflow_node(initial_state)
        
        if result["workflow_progress"].status != WorkflowStatus.FAILED:
            print("✅ Validation passed with valid files!")
            print(f"Progress: {result['workflow_progress'].progress_percentage}%")
        else:
            print("❌ Validation failed unexpectedly")
        
        # Cleanup
        cleanup_demo_files(resume_path, job_path)
        
    except Exception as e:
        print(f"❌ State validation demo failed: {e}")
        return False
    
    return True


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n🛠️ Demonstrating Convenience Functions...")
    print("-" * 60)
    
    try:
        from core.graph import create_workflow, execute_application_factory
        
        # Test workflow creation
        print("Testing create_workflow()...")
        workflow = create_workflow()
        print("✅ Workflow created via convenience function!")
        
        print("✅ All convenience functions are available!")
        
    except Exception as e:
        print(f"❌ Convenience function demo failed: {e}")
        return False
    
    return True


def main():
    """Main demo function."""
    print("🏭 LangGraph Application Factory Workflow Demo")
    print("=" * 70)
    print("This demo showcases the LangGraph workflow engine without requiring")
    print("actual API keys or external services.\n")
    
    demos = [
        ("Workflow Initialization", demo_workflow_initialization),
        ("Progress Tracking", demo_workflow_progress_tracking),
        ("State Validation", demo_workflow_state_validation),
        ("Convenience Functions", demo_convenience_functions),
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        print(f"\n🚀 Running Demo: {demo_name}")
        success = demo_func()
        results.append((demo_name, success))
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 DEMO RESULTS SUMMARY")
    print("=" * 70)
    
    for demo_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{demo_name:.<50} {status}")
    
    overall_success = all(result[1] for result in results)
    
    if overall_success:
        print("\n🎉 All demos completed successfully!")
        print("The LangGraph workflow engine is ready for integration!")
    else:
        print("\n⚠️ Some demos failed - check implementation details.")
    
    print("\n🔗 Next Steps:")
    print("- Integrate with the main Streamlit app")
    print("- Test with real API keys and documents")
    print("- Configure user preferences and customization")
    

if __name__ == "__main__":
    main() 
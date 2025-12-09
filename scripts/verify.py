#!/usr/bin/env python3
"""
Verify GRAIL-LM implementation completeness.
Checks that all key components are in place.
"""
import os
import sys
from pathlib import Path

def check_file(path: str, description: str) -> bool:
    """Check if a file exists"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def check_import(module: str, description: str) -> bool:
    """Check if a module can be imported"""
    try:
        __import__(module)
        print(f"✅ {description}: {module}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module} - {e}")
        return False

def main():
    print("🔍 GRAIL-LM Implementation Verification\n")
    
    checks = []
    
    print("📁 Core Files:")
    checks.append(check_file("backend/services/neo4j_store.py", "Neo4j Integration"))
    checks.append(check_file("backend/services/gnn_retriever.py", "GNN Retriever"))
    checks.append(check_file("backend/services/rag.py", "RAG Pipeline"))
    checks.append(check_file("backend/services/paths.py", "Path Finding"))
    checks.append(check_file("app/streamlit_app.py", "Streamlit Dashboard"))
    checks.append(check_file("docker-compose.yml", "Docker Compose"))
    checks.append(check_file("README.md", "Documentation"))
    
    print("\n🛠️ Scripts:")
    checks.append(check_file("scripts/load_neo4j.py", "Neo4j Loader"))
    checks.append(check_file("scripts/train_gnn.py", "GNN Training"))
    checks.append(check_file("scripts/setup.ps1", "Setup Script"))
    
    print("\n📦 Configuration:")
    checks.append(check_file("requirements.txt", "Requirements"))
    checks.append(check_file(".env.example", "Environment Template"))
    checks.append(check_file("IMPLEMENTATION_SUMMARY.md", "Implementation Summary"))
    
    print("\n🧪 Python Imports:")
    checks.append(check_import("backend.services.neo4j_store", "Neo4j Module"))
    checks.append(check_import("backend.services.gnn_retriever", "GNN Module"))
    checks.append(check_import("backend.services.rag", "RAG Module"))
    checks.append(check_import("backend.api", "API Module"))
    
    print("\n📊 Summary:")
    passed = sum(checks)
    total = len(checks)
    percentage = (passed / total) * 100
    
    print(f"Passed: {passed}/{total} ({percentage:.1f}%)")
    
    if percentage == 100:
        print("\n✅ All checks passed! GRAIL-LM is ready.")
        return 0
    elif percentage >= 80:
        print("\n⚠️  Most checks passed. Review failed items.")
        return 1
    else:
        print("\n❌ Many checks failed. Please review implementation.")
        return 2

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Database setup script for Neon PostgreSQL integration
This script creates the required tables in the Neon database
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def setup_database():
    """
    Setup the Neon database tables
    """
    print("Setting up Neon database tables...")

    try:
        # Import after adding to path
        from utils.db_connection import db_manager
        from models.document_model import Base

        # Check if database is configured
        if not db_manager.engine:
            print("[ERROR] Neon database not configured in environment variables")
            print("   Please ensure DATABASE_URL is set in your .env file")
            return False

        # Create all tables with checkfirst=True to avoid duplicate table errors
        Base.metadata.create_all(bind=db_manager.engine, checkfirst=True)
        print("[SUCCESS] Database tables verified/created successfully!")

        # Test the connection
        if db_manager.test_connection():
            print("[SUCCESS] Database connection test passed")
        else:
            print("[WARNING] Database connection test failed")

        return True

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error setting up database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = setup_database()
    if success:
        print("\n[SUCCESS] Neon database setup completed successfully!")
        print("   The system will now store document metadata in the Neon database")
        print("   while maintaining vector embeddings in Qdrant for optimal performance.")
    else:
        print("\n[ERROR] Database setup failed. Please check your configuration.")
        sys.exit(1)
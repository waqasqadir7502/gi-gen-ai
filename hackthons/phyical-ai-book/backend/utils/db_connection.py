from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Import config and logger
from config import config
from utils.logger import log_info, log_error
import urllib.parse

class DatabaseManager:
    """
    Database manager for Neon PostgreSQL connection
    """
    def __init__(self):
        if config.DATABASE_URL:
            # Properly encode the database URL to handle special characters
            self.engine = create_engine(
                config.DATABASE_URL,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=300,    # Recycle connections every 5 minutes
                echo=False           # Set to True for SQL debugging
            )

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            log_info("Database engine initialized", extra={
                "database_url": f"postgresql://{config.DATABASE_URL.split('@')[1].split('/')[0]}/neondb" if config.DATABASE_URL else "None"
            })
        else:
            self.engine = None
            self.SessionLocal = None
            log_error("DATABASE_URL not configured in environment")

    def get_session(self):
        """
        Get a database session
        """
        if not self.SessionLocal:
            raise ValueError("Database not properly configured")

        return self.SessionLocal()

    def test_connection(self):
        """
        Test database connection
        """
        if not self.engine:
            return False

        try:
            with self.engine.connect() as conn:
                # Execute a simple query to test connection
                result = conn.execute(text("SELECT 1"))
                return result.fetchone()[0] == 1
        except Exception as e:
            log_error(f"Database connection test failed: {str(e)}")
            return False

    def create_tables(self):
        """
        Create all tables defined in models
        """
        if not self.engine:
            raise ValueError("Database not properly configured")

        try:
            from ..models.document_model import Base
            Base.metadata.create_all(bind=self.engine)
            log_info("Database tables created successfully")
            return True
        except Exception as e:
            log_error(f"Error creating database tables: {str(e)}")
            return False

    def close_engine(self):
        """
        Close the database engine
        """
        if self.engine:
            self.engine.dispose()

# Create a singleton instance
db_manager = DatabaseManager()
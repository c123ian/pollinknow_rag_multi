# database/db_operations.py
import os
import sqlite3
import logging
import datetime
from typing import Dict, List, Any, Optional

from config import DATABASE_DIR, DB_PATH

class DatabaseManager:
    def __init__(self):
        """Initialize database connection and create tables if needed"""
        # Ensure directory exists
        os.makedirs(DATABASE_DIR, exist_ok=True)
        
        try:
            # Connect to database
            self.conn = sqlite3.connect(DB_PATH)
            self.cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            self._create_tables()
            
        except Exception as e:
            logging.error(f"Database initialization error: {e}")
            raise RuntimeError(f"Failed to initialize database: {str(e)}")
            
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        # Check if image_analyses table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='image_analyses'")
        if not self.cursor.fetchone():
            # Create the table
            self.cursor.execute('''
                CREATE TABLE image_analyses (
                    analysis_id TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT,
                    context_source TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
            logging.info("Created image_analyses table")
        else:
            # Check if context_source column exists, add it if not
            self.cursor.execute("PRAGMA table_info(image_analyses)")
            columns = [column[1] for column in self.cursor.fetchall()]
            if 'context_source' not in columns:
                self.cursor.execute('ALTER TABLE image_analyses ADD COLUMN context_source TEXT')
                self.conn.commit()
                logging.info("Added context_source column to image_analyses table")
    
    async def save_analysis(self, analysis_id: str, image_path: str, 
                           analysis_type: str, query: str, response: str, 
                           top_sources: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Save analysis results to the database
        
        Args:
            analysis_id: Unique identifier for the analysis
            image_path: Path to the analyzed image
            analysis_type: Type of analysis (classification, etc.)
            query: The query used
            response: Model response
            top_sources: Optional list of top source documents
            
        Returns:
            True if saved successfully, False otherwise
            
        Raises:
            ValueError: If required parameters are missing
        """
        if not analysis_id or not image_path or not analysis_type or not query:
            raise ValueError("Missing required parameters for database save")
            
        try:
            # Insert basic info
            self.cursor.execute(
                "INSERT INTO image_analyses (analysis_id, image_path, analysis_type, query, response) VALUES (?, ?, ?, ?, ?)",
                (analysis_id, image_path, analysis_type, query, response)
            )
            
            # Add context source if available
            if top_sources and len(top_sources) > 0:
                top_source = top_sources[0]
                context_source = f"{top_source['filename']} (page {top_source['page']})"
                
                self.cursor.execute(
                    "UPDATE image_analyses SET context_source = ? WHERE analysis_id = ?",
                    (context_source, analysis_id)
                )
            
            self.conn.commit()
            logging.info(f"Saved analysis {analysis_id} to database")
            return True
            
        except Exception as e:
            logging.error(f"Database error saving analysis: {e}")
            # Rollback transaction
            self.conn.rollback()
            return False
            
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
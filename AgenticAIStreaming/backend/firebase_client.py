"""
Firebase Admin SDK client for Agent Server
Handles writing streaming responses back to Firestore
"""
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import asyncio

load_dotenv()


class FirebaseClient:
    """Firebase Admin SDK client for writing responses to Firestore"""

    def __init__(self):
        self.project_id = os.getenv("FIREBASE_PROJECT_ID", "")
        self.service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "")
        self._initialized = False

    def _initialize(self):
        """Initialize Firebase Admin SDK"""
        if self._initialized:
            return

        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            if not firebase_admin._apps:
                if self.service_account_path and os.path.exists(
                    self.service_account_path
                ):
                    cred = credentials.Certificate(self.service_account_path)
                    firebase_admin.initialize_app(cred)
                else:
                    # Use default credentials (e.g., from environment)
                    firebase_admin.initialize_app()

            self.db = firestore.client()
            self.firestore = firestore  # Store for SERVER_TIMESTAMP
            self._initialized = True
        except ImportError:
            raise ImportError(
                "firebase-admin is required. Install it with: pip install firebase-admin"
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Firebase: {str(e)}")

    async def write_response_chunk(
        self,
        conversation_id: str,
        message_id: str,
        chunk: str,
        is_complete: bool = False,
    ) -> None:
        """
        Write a response chunk to Firestore

        Args:
            conversation_id: Conversation identifier
            message_id: Message identifier
            chunk: Response chunk text
            is_complete: Whether this is the final chunk
        """
        if not self._initialized:
            self._initialize()

        try:
            # Write to aiResponses collection
            response_ref = self.db.collection("conversations").document(
                conversation_id
            ).collection("aiResponses").document(message_id)

            # Get current data
            doc = response_ref.get()
            if doc.exists:
                current_data = doc.to_dict()
                current_text = current_data.get("currentText", "")
                chunks = current_data.get("chunks", [])
            else:
                current_text = ""
                chunks = []

            # Append new chunk
            new_text = current_text + chunk
            chunks.append(chunk)

            # Update document
            response_ref.set(
                {
                    "currentText": new_text,
                    "chunks": chunks,
                    "isComplete": is_complete,
                    "updatedAt": self.firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
        except Exception as e:
            raise Exception(f"Failed to write chunk to Firebase: {str(e)}")

    async def write_complete_response(
        self,
        conversation_id: str,
        message_id: str,
        full_response: str,
    ) -> None:
        """
        Write a complete response to Firestore

        Args:
            conversation_id: Conversation identifier
            message_id: Message identifier
            full_response: Complete response text
        """
        if not self._initialized:
            self._initialize()

        try:
            # Write to aiResponses collection
            response_ref = self.db.collection("conversations").document(
                conversation_id
            ).collection("aiResponses").document(message_id)

            response_ref.set(
                {
                    "currentText": full_response,
                    "chunks": [full_response],
                    "isComplete": True,
                    "updatedAt": self.firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )

            # Also create a message in the messages collection for the assistant
            messages_ref = self.db.collection("conversations").document(
                conversation_id
            ).collection("messages")

            messages_ref.add(
                {
                    "text": full_response,
                    "senderId": "assistant",
                    "timestamp": self.firestore.SERVER_TIMESTAMP,
                    "type": "text",
                    "messageId": message_id,
                }
            )
        except Exception as e:
            raise Exception(f"Failed to write complete response to Firebase: {str(e)}")

    def is_configured(self) -> bool:
        """Check if Firebase is configured"""
        return bool(self.project_id or self.service_account_path)


# Singleton instance
_firebase_client: Optional[FirebaseClient] = None


def get_firebase_client() -> Optional[FirebaseClient]:
    """Get or create Firebase client instance"""
    global _firebase_client
    if _firebase_client is None:
        _firebase_client = FirebaseClient()
        if _firebase_client.is_configured():
            try:
                _firebase_client._initialize()
            except Exception:
                # Firebase not properly configured, return None
                _firebase_client = None
    return _firebase_client


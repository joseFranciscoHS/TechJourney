"""
Chat21 integration for Python backend

Chat21 is built on Firebase Cloud Functions. This module provides a Python client
to interact with Chat21's Firebase Cloud Functions API.

Reference: https://github.com/chat21/chat21-cloud-functions
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import httpx

load_dotenv()


class Chat21Config:
    """Chat21 configuration for Firebase Cloud Functions"""

    def __init__(self):
        # Firebase project configuration
        self.project_id = os.getenv("FIREBASE_PROJECT_ID", "")
        self.region = os.getenv("FIREBASE_FUNCTIONS_REGION", "us-central1")

        # Firebase Admin SDK credentials (path to service account JSON)
        self.service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "")

        # Chat21 Cloud Functions base URL
        # Format: https://<region>-<project-id>.cloudfunctions.net
        if self.project_id:
            self.base_url = (
                f"https://{self.region}-{self.project_id}.cloudfunctions.net"
            )
        else:
            self.base_url = os.getenv("CHAT21_FUNCTIONS_URL", "")

    def is_configured(self) -> bool:
        """Check if Chat21 is properly configured"""
        return bool(self.project_id or self.base_url)


class Chat21Client:
    """
    Chat21 API client for Firebase Cloud Functions

    Chat21 uses Firebase Cloud Functions deployed at /api and /supportapi endpoints.
    Authentication is done via Firebase JWT tokens.

    To use this client:
    1. Deploy Chat21 Cloud Functions to your Firebase project
    2. Set FIREBASE_PROJECT_ID in environment variables
    3. Use Firebase Admin SDK to create custom tokens or verify ID tokens
    """

    def __init__(self, config: Optional[Chat21Config] = None):
        self.config = config or Chat21Config()
        self.client = httpx.AsyncClient(base_url=self.config.base_url, timeout=30.0)

    async def send_message(
        self,
        jwt_token: str,
        recipient_id: str,
        message: str,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a message via Chat21 Cloud Functions

        Args:
            jwt_token: Firebase JWT authentication token
            recipient_id: Recipient user ID
            message: Message content
            message_type: Type of message (text, image, etc.)
            metadata: Optional metadata for the message

        Returns:
            Response from Chat21 API

        Reference: https://github.com/chat21/chat21-cloud-functions/blob/master/docs/api.md
        """
        if not self.config.is_configured():
            raise ValueError(
                "Chat21 is not configured. Set FIREBASE_PROJECT_ID or CHAT21_FUNCTIONS_URL"
            )

        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
        }

        payload = {"recipient": recipient_id, "text": message, "type": message_type}

        if metadata:
            payload["metadata"] = metadata

        try:
            # Chat21 sends messages via /api/messages endpoint
            response = await self.client.post(
                "/api/messages", headers=headers, json=payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise Exception(f"Chat21 API error: {str(e)}")

    async def get_conversation(
        self, jwt_token: str, conversation_id: str
    ) -> Dict[str, Any]:
        """
        Get conversation details

        Args:
            jwt_token: Firebase JWT authentication token
            conversation_id: Conversation identifier

        Returns:
            Conversation data
        """
        if not self.config.is_configured():
            raise ValueError("Chat21 is not configured")

        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
        }

        try:
            response = await self.client.get(
                f"/api/conversations/{conversation_id}", headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise Exception(f"Chat21 API error: {str(e)}")

    async def create_custom_token(
        self, user_id: str, additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a Firebase custom token for authentication

        This requires Firebase Admin SDK to be initialized.
        The token can then be used to authenticate with Chat21.

        Args:
            user_id: User identifier
            additional_claims: Optional additional claims

        Returns:
            Custom token string
        """
        try:
            import firebase_admin
            from firebase_admin import auth, credentials

            # Initialize Firebase Admin if not already initialized
            if not firebase_admin._apps:
                if self.config.service_account_path:
                    cred = credentials.Certificate(self.config.service_account_path)
                    firebase_admin.initialize_app(cred)
                else:
                    # Use default credentials (e.g., from environment)
                    firebase_admin.initialize_app()

            # Create custom token
            custom_token = auth.create_custom_token(user_id, additional_claims or {})
            return custom_token.decode("utf-8")
        except ImportError:
            raise ImportError(
                "firebase-admin is required. Install it with: pip install firebase-admin"
            )
        except Exception as e:
            raise Exception(f"Failed to create custom token: {str(e)}")

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Singleton instance
_chat21_client: Optional[Chat21Client] = None


def get_chat21_client() -> Optional[Chat21Client]:
    """Get or create Chat21 client instance"""
    global _chat21_client
    if _chat21_client is None:
        config = Chat21Config()
        if config.is_configured():
            _chat21_client = Chat21Client(config)
    return _chat21_client

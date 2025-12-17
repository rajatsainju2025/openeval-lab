"""Collaboration hub for multi-user explanation features.

This module provides functionality for team collaboration on code explanations,
including shared sessions, real-time updates, and collaborative editing.

Example:
    >>> from openeval.explainers import CollaborationHub, create_session
    >>> hub = get_collaboration_hub()
    >>> session = hub.create_session("code-review", owner="user1")
    >>> hub.join_session(session.session_id, "user2")
    >>> hub.share_explanation(session.session_id, explanation)
"""

from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable


class SessionStatus(Enum):
    """Status of a collaboration session."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ParticipantRole(Enum):
    """Roles for session participants."""

    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"
    REVIEWER = "reviewer"


class ActivityType(Enum):
    """Types of collaboration activities."""

    JOIN = "join"
    LEAVE = "leave"
    COMMENT = "comment"
    EDIT = "edit"
    APPROVE = "approve"
    REJECT = "reject"
    SHARE = "share"
    REACTION = "reaction"


class SharePermission(Enum):
    """Permissions for shared content."""

    VIEW = "view"
    COMMENT = "comment"
    EDIT = "edit"
    ADMIN = "admin"


@dataclass
class Participant:
    """A participant in a collaboration session."""

    user_id: str
    username: str
    role: ParticipantRole
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    is_online: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_active = datetime.now()


@dataclass
class Activity:
    """An activity record in a session."""

    activity_id: str
    activity_type: ActivityType
    user_id: str
    timestamp: datetime
    content: dict[str, Any]
    target_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Comment:
    """A comment on shared content."""

    comment_id: str
    user_id: str
    content: str
    target_id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    replies: list[Comment] = field(default_factory=list)
    reactions: dict[str, list[str]] = field(default_factory=dict)
    resolved: bool = False

    def add_reply(self, user_id: str, content: str) -> Comment:
        """Add a reply to this comment."""
        reply = Comment(
            comment_id=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            target_id=self.comment_id,
        )
        self.replies.append(reply)
        return reply

    def add_reaction(self, user_id: str, reaction: str) -> None:
        """Add a reaction to this comment."""
        if reaction not in self.reactions:
            self.reactions[reaction] = []
        if user_id not in self.reactions[reaction]:
            self.reactions[reaction].append(user_id)


@dataclass
class SharedContent:
    """Content shared in a session."""

    content_id: str
    content_type: str
    content: Any
    owner_id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    permissions: dict[str, SharePermission] = field(default_factory=dict)
    comments: list[Comment] = field(default_factory=list)
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_comment(self, user_id: str, content: str) -> Comment:
        """Add a comment to this content."""
        comment = Comment(
            comment_id=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            target_id=self.content_id,
        )
        self.comments.append(comment)
        return comment

    def update_content(self, new_content: Any, user_id: str) -> None:
        """Update the content."""
        self.content = new_content
        self.updated_at = datetime.now()
        self.version += 1
        self.metadata["last_editor"] = user_id


@dataclass
class CollaborationSession:
    """A collaboration session."""

    session_id: str
    name: str
    owner_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    participants: dict[str, Participant] = field(default_factory=dict)
    shared_content: dict[str, SharedContent] = field(default_factory=dict)
    activities: list[Activity] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def participant_count(self) -> int:
        """Get number of participants."""
        return len(self.participants)

    @property
    def online_count(self) -> int:
        """Get number of online participants."""
        return sum(1 for p in self.participants.values() if p.is_online)

    def add_participant(
        self,
        user_id: str,
        username: str,
        role: ParticipantRole = ParticipantRole.VIEWER,
    ) -> Participant:
        """Add a participant to the session."""
        participant = Participant(
            user_id=user_id,
            username=username,
            role=role,
        )
        self.participants[user_id] = participant
        self._record_activity(ActivityType.JOIN, user_id, {})
        return participant

    def remove_participant(self, user_id: str) -> bool:
        """Remove a participant from the session."""
        if user_id in self.participants:
            del self.participants[user_id]
            self._record_activity(ActivityType.LEAVE, user_id, {})
            return True
        return False

    def share_content(
        self,
        content_id: str,
        content_type: str,
        content: Any,
        owner_id: str,
    ) -> SharedContent:
        """Share content in the session."""
        shared = SharedContent(
            content_id=content_id,
            content_type=content_type,
            content=content,
            owner_id=owner_id,
        )
        self.shared_content[content_id] = shared
        self._record_activity(ActivityType.SHARE, owner_id, {"content_id": content_id})
        return shared

    def _record_activity(
        self,
        activity_type: ActivityType,
        user_id: str,
        content: dict[str, Any],
    ) -> Activity:
        """Record an activity."""
        activity = Activity(
            activity_id=str(uuid.uuid4()),
            activity_type=activity_type,
            user_id=user_id,
            timestamp=datetime.now(),
            content=content,
        )
        self.activities.append(activity)
        self.updated_at = datetime.now()
        return activity


@dataclass
class SessionInvite:
    """An invitation to join a session."""

    invite_id: str
    session_id: str
    inviter_id: str
    invitee_email: str
    role: ParticipantRole
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=7))
    accepted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if invite is expired."""
        return datetime.now() > self.expires_at


class CollaborationEventHandler(ABC):
    """Abstract handler for collaboration events."""

    @abstractmethod
    def on_participant_join(self, session: CollaborationSession, participant: Participant) -> None:
        """Handle participant joining."""
        pass

    @abstractmethod
    def on_participant_leave(self, session: CollaborationSession, user_id: str) -> None:
        """Handle participant leaving."""
        pass

    @abstractmethod
    def on_content_shared(self, session: CollaborationSession, content: SharedContent) -> None:
        """Handle content being shared."""
        pass

    @abstractmethod
    def on_comment_added(self, session: CollaborationSession, comment: Comment) -> None:
        """Handle comment being added."""
        pass


class DefaultEventHandler(CollaborationEventHandler):
    """Default event handler that logs events."""

    def on_participant_join(self, session: CollaborationSession, participant: Participant) -> None:
        """Log participant join."""
        pass

    def on_participant_leave(self, session: CollaborationSession, user_id: str) -> None:
        """Log participant leave."""
        pass

    def on_content_shared(self, session: CollaborationSession, content: SharedContent) -> None:
        """Log content shared."""
        pass

    def on_comment_added(self, session: CollaborationSession, comment: Comment) -> None:
        """Log comment added."""
        pass


class CollaborationStorage(ABC):
    """Abstract storage for collaboration data."""

    @abstractmethod
    def save_session(self, session: CollaborationSession) -> None:
        """Save a session."""
        pass

    @abstractmethod
    def load_session(self, session_id: str) -> CollaborationSession | None:
        """Load a session."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        pass

    @abstractmethod
    def list_sessions(self, user_id: str | None = None) -> list[CollaborationSession]:
        """List sessions."""
        pass


class InMemoryCollaborationStorage(CollaborationStorage):
    """In-memory storage for collaboration data."""

    def __init__(self) -> None:
        """Initialize storage."""
        self._sessions: dict[str, CollaborationSession] = {}
        self._lock = threading.Lock()

    def save_session(self, session: CollaborationSession) -> None:
        """Save a session."""
        with self._lock:
            self._sessions[session.session_id] = session

    def load_session(self, session_id: str) -> CollaborationSession | None:
        """Load a session."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self, user_id: str | None = None) -> list[CollaborationSession]:
        """List sessions."""
        sessions = list(self._sessions.values())
        if user_id:
            sessions = [s for s in sessions if user_id in s.participants or s.owner_id == user_id]
        return sessions


class CollaborationHub:
    """Main hub for collaboration features."""

    def __init__(
        self,
        storage: CollaborationStorage | None = None,
        event_handler: CollaborationEventHandler | None = None,
    ) -> None:
        """Initialize the collaboration hub.

        Args:
            storage: Storage backend for sessions.
            event_handler: Handler for collaboration events.
        """
        self.storage = storage or InMemoryCollaborationStorage()
        self.event_handler = event_handler or DefaultEventHandler()
        self._subscriptions: dict[str, list[Callable]] = defaultdict(list)
        self._invites: dict[str, SessionInvite] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        name: str,
        owner_id: str,
        owner_username: str = "",
        settings: dict[str, Any] | None = None,
    ) -> CollaborationSession:
        """Create a new collaboration session.

        Args:
            name: Session name.
            owner_id: ID of the session owner.
            owner_username: Username of the owner.
            settings: Session settings.

        Returns:
            The created session.
        """
        session = CollaborationSession(
            session_id=str(uuid.uuid4()),
            name=name,
            owner_id=owner_id,
            settings=settings or {},
        )

        # Add owner as first participant
        session.add_participant(
            owner_id,
            owner_username or owner_id,
            ParticipantRole.OWNER,
        )

        self.storage.save_session(session)
        return session

    def get_session(self, session_id: str) -> CollaborationSession | None:
        """Get a session by ID.

        Args:
            session_id: Session ID.

        Returns:
            The session if found.
        """
        return self.storage.load_session(session_id)

    def join_session(
        self,
        session_id: str,
        user_id: str,
        username: str = "",
        role: ParticipantRole = ParticipantRole.VIEWER,
    ) -> Participant | None:
        """Join an existing session.

        Args:
            session_id: Session ID to join.
            user_id: User ID.
            username: Username.
            role: Role in the session.

        Returns:
            Participant object if successful.
        """
        session = self.storage.load_session(session_id)
        if not session:
            return None

        if session.status != SessionStatus.ACTIVE:
            return None

        participant = session.add_participant(user_id, username or user_id, role)
        self.storage.save_session(session)
        self.event_handler.on_participant_join(session, participant)
        self._notify_subscribers(session_id, "participant_joined", participant)

        return participant

    def leave_session(self, session_id: str, user_id: str) -> bool:
        """Leave a session.

        Args:
            session_id: Session ID.
            user_id: User ID.

        Returns:
            True if successful.
        """
        session = self.storage.load_session(session_id)
        if not session:
            return False

        if session.remove_participant(user_id):
            self.storage.save_session(session)
            self.event_handler.on_participant_leave(session, user_id)
            self._notify_subscribers(session_id, "participant_left", user_id)
            return True

        return False

    def share_explanation(
        self,
        session_id: str,
        explanation: Any,
        owner_id: str,
        content_id: str | None = None,
    ) -> SharedContent | None:
        """Share an explanation in a session.

        Args:
            session_id: Session ID.
            explanation: The explanation to share.
            owner_id: Owner's user ID.
            content_id: Optional content ID.

        Returns:
            SharedContent if successful.
        """
        session = self.storage.load_session(session_id)
        if not session:
            return None

        content_id = content_id or str(uuid.uuid4())
        content = session.share_content(
            content_id=content_id,
            content_type="explanation",
            content=explanation,
            owner_id=owner_id,
        )

        self.storage.save_session(session)
        self.event_handler.on_content_shared(session, content)
        self._notify_subscribers(session_id, "content_shared", content)

        return content

    def add_comment(
        self,
        session_id: str,
        content_id: str,
        user_id: str,
        comment_text: str,
    ) -> Comment | None:
        """Add a comment to shared content.

        Args:
            session_id: Session ID.
            content_id: Content ID to comment on.
            user_id: User ID.
            comment_text: Comment text.

        Returns:
            Comment if successful.
        """
        session = self.storage.load_session(session_id)
        if not session:
            return None

        content = session.shared_content.get(content_id)
        if not content:
            return None

        comment = content.add_comment(user_id, comment_text)
        session._record_activity(
            ActivityType.COMMENT,
            user_id,
            {"content_id": content_id, "comment_id": comment.comment_id},
        )

        self.storage.save_session(session)
        self.event_handler.on_comment_added(session, comment)
        self._notify_subscribers(session_id, "comment_added", comment)

        return comment

    def create_invite(
        self,
        session_id: str,
        inviter_id: str,
        invitee_email: str,
        role: ParticipantRole = ParticipantRole.VIEWER,
    ) -> SessionInvite | None:
        """Create an invitation to join a session.

        Args:
            session_id: Session ID.
            inviter_id: Inviter's user ID.
            invitee_email: Invitee's email.
            role: Role to assign.

        Returns:
            SessionInvite if successful.
        """
        session = self.storage.load_session(session_id)
        if not session:
            return None

        # Check if inviter has permission
        inviter = session.participants.get(inviter_id)
        if not inviter or inviter.role not in [
            ParticipantRole.OWNER,
            ParticipantRole.EDITOR,
        ]:
            return None

        invite = SessionInvite(
            invite_id=str(uuid.uuid4()),
            session_id=session_id,
            inviter_id=inviter_id,
            invitee_email=invitee_email,
            role=role,
        )

        with self._lock:
            self._invites[invite.invite_id] = invite

        return invite

    def accept_invite(
        self,
        invite_id: str,
        user_id: str,
        username: str = "",
    ) -> Participant | None:
        """Accept an invitation.

        Args:
            invite_id: Invite ID.
            user_id: User ID accepting.
            username: Username.

        Returns:
            Participant if successful.
        """
        invite = self._invites.get(invite_id)
        if not invite or invite.is_expired or invite.accepted:
            return None

        participant = self.join_session(
            invite.session_id,
            user_id,
            username,
            invite.role,
        )

        if participant:
            invite.accepted = True

        return participant

    def subscribe(
        self,
        session_id: str,
        callback: Callable[[str, str, Any], None],
    ) -> str:
        """Subscribe to session events.

        Args:
            session_id: Session ID to subscribe to.
            callback: Callback function (session_id, event, data).

        Returns:
            Subscription ID.
        """
        subscription_id = str(uuid.uuid4())
        with self._lock:
            self._subscriptions[session_id].append((subscription_id, callback))
        return subscription_id

    def unsubscribe(self, session_id: str, subscription_id: str) -> bool:
        """Unsubscribe from session events.

        Args:
            session_id: Session ID.
            subscription_id: Subscription ID.

        Returns:
            True if successful.
        """
        with self._lock:
            subs = self._subscriptions.get(session_id, [])
            self._subscriptions[session_id] = [
                (sid, cb) for sid, cb in subs if sid != subscription_id
            ]
        return True

    def _notify_subscribers(
        self,
        session_id: str,
        event: str,
        data: Any,
    ) -> None:
        """Notify all subscribers of an event."""
        with self._lock:
            subs = self._subscriptions.get(session_id, []).copy()

        for _, callback in subs:
            try:
                callback(session_id, event, data)
            except Exception:
                pass

    def get_user_sessions(self, user_id: str) -> list[CollaborationSession]:
        """Get all sessions for a user.

        Args:
            user_id: User ID.

        Returns:
            List of sessions.
        """
        return self.storage.list_sessions(user_id)

    def close_session(self, session_id: str, user_id: str) -> bool:
        """Close a session.

        Args:
            session_id: Session ID.
            user_id: User requesting closure.

        Returns:
            True if successful.
        """
        session = self.storage.load_session(session_id)
        if not session:
            return False

        if session.owner_id != user_id:
            return False

        session.status = SessionStatus.COMPLETED
        self.storage.save_session(session)
        self._notify_subscribers(session_id, "session_closed", {})
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get hub statistics.

        Returns:
            Statistics dictionary.
        """
        all_sessions = self.storage.list_sessions()
        active = sum(1 for s in all_sessions if s.status == SessionStatus.ACTIVE)
        total_participants = sum(s.participant_count for s in all_sessions)
        total_content = sum(len(s.shared_content) for s in all_sessions)

        return {
            "total_sessions": len(all_sessions),
            "active_sessions": active,
            "total_participants": total_participants,
            "total_shared_content": total_content,
            "pending_invites": len(
                [i for i in self._invites.values() if not i.accepted and not i.is_expired]
            ),
        }


# Global instance
_collaboration_hub: CollaborationHub | None = None


def get_collaboration_hub() -> CollaborationHub:
    """Get the global collaboration hub.

    Returns:
        The global CollaborationHub instance.
    """
    global _collaboration_hub
    if _collaboration_hub is None:
        _collaboration_hub = CollaborationHub()
    return _collaboration_hub


def reset_collaboration_hub() -> None:
    """Reset the global collaboration hub."""
    global _collaboration_hub
    _collaboration_hub = None


def create_collaboration_hub(
    storage: CollaborationStorage | None = None,
    event_handler: CollaborationEventHandler | None = None,
) -> CollaborationHub:
    """Create a new collaboration hub.

    Args:
        storage: Storage backend.
        event_handler: Event handler.

    Returns:
        New CollaborationHub instance.
    """
    return CollaborationHub(storage=storage, event_handler=event_handler)


def create_session(
    name: str,
    owner_id: str,
    **kwargs: Any,
) -> CollaborationSession:
    """Create a new collaboration session.

    Args:
        name: Session name.
        owner_id: Owner's user ID.
        **kwargs: Additional parameters.

    Returns:
        The created session.
    """
    return get_collaboration_hub().create_session(name, owner_id, **kwargs)


def join_session(
    session_id: str,
    user_id: str,
    **kwargs: Any,
) -> Participant | None:
    """Join an existing session.

    Args:
        session_id: Session ID.
        user_id: User ID.
        **kwargs: Additional parameters.

    Returns:
        Participant if successful.
    """
    return get_collaboration_hub().join_session(session_id, user_id, **kwargs)


def share_in_session(
    session_id: str,
    content: Any,
    owner_id: str,
) -> SharedContent | None:
    """Share content in a session.

    Args:
        session_id: Session ID.
        content: Content to share.
        owner_id: Owner's user ID.

    Returns:
        SharedContent if successful.
    """
    return get_collaboration_hub().share_explanation(session_id, content, owner_id)

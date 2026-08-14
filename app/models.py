"""SQLAlchemy 2.0 ORM 模型。"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32))
    total: Mapped[float] = mapped_column(Float)
    created_at: Mapped[str] = mapped_column(String(32))
    shipped_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    tracking_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    carrier: Mapped[str | None] = mapped_column(String(64), nullable=True)
    estimated_delivery: Mapped[str | None] = mapped_column(String(32), nullable=True)
    delivered_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    refund_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    refund_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    refund_requested_at: Mapped[str | None] = mapped_column(String(32), nullable=True)

    items: Mapped[list["OrderItem"]] = relationship(
        "OrderItem", back_populates="order", lazy="raise"
    )


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.order_id"))
    name: Mapped[str] = mapped_column(String(256))
    sku: Mapped[str] = mapped_column(String(64))
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)

    order: Mapped["Order"] = relationship("Order", back_populates="items")


class UserProfile(Base):
    """用户画像 key-value 存储。"""

    __tablename__ = "user_profiles"
    __table_args__ = (UniqueConstraint("user_id", "key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    key: Mapped[str] = mapped_column(String(128))
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MemoryEmbedding(Base):
    """长期记忆向量存储 (embedding-1536)。"""

    __tablename__ = "memory_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
    importance: Mapped[float] = mapped_column(Float, default=1.0)
    category: Mapped[str] = mapped_column(String(64), default="other")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class User(Base):
    """用户模型，存储认证信息。"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

"""merge 4e43a6449cf4 and 85fbdd72c09b

Revision ID: 443fbae4c826
Revises: 4e43a6449cf4, 85fbdd72c09b
Create Date: 2020-12-02 16:23:43.199892

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '443fbae4c826'
down_revision = ('4e43a6449cf4', '85fbdd72c09b')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass

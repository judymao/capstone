"""empty message

Revision ID: dc70bfa70bd7
Revises: 2a6dcf8b0126
Create Date: 2020-11-23 22:16:13.407894

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'dc70bfa70bd7'
down_revision = '2a6dcf8b0126'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('portfolio_info', sa.Column('cash', sa.Float(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('portfolio_info', 'cash')
    # ### end Alembic commands ###
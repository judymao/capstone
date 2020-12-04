"""empty message

Revision ID: 6bc7d2e53f13
Revises: 443fbae4c826
Create Date: 2020-12-03 08:48:42.592836

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6bc7d2e53f13'
down_revision = '443fbae4c826'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('portfolio_info', sa.Column('risk_appetite', sa.String(length=255), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('portfolio_info', 'risk_appetite')
    # ### end Alembic commands ###

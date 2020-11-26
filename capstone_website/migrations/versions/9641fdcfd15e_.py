"""empty message

Revision ID: 9641fdcfd15e
Revises: b64d2caf2a7b
Create Date: 2020-11-24 16:56:51.850145

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9641fdcfd15e'
down_revision = 'b64d2caf2a7b'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('portfolio_info', sa.Column('cash', sa.Float(), nullable=True))
    op.add_column('portfolio_info', sa.Column('games_philosophy', sa.Float(), nullable=True))
    op.add_column('portfolio_info', sa.Column('job_philosophy', sa.Float(), nullable=True))
    op.add_column('portfolio_info', sa.Column('lose_philosophy', sa.Float(), nullable=True))
    op.add_column('portfolio_info', sa.Column('monitor_philosophy', sa.Float(), nullable=True))
    op.add_column('portfolio_info', sa.Column('unknown_philosophy', sa.Float(), nullable=True))
    op.add_column('portfolio_info', sa.Column('win_philosophy', sa.Float(), nullable=True))
    op.drop_column('portfolio_info', 'protect_portfolio')
    op.drop_column('portfolio_info', 'next_expenditure')
    op.drop_column('portfolio_info', 'inv_philosophy')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('portfolio_info', sa.Column('inv_philosophy', sa.VARCHAR(length=255), autoincrement=False, nullable=True))
    op.add_column('portfolio_info', sa.Column('next_expenditure', sa.VARCHAR(length=255), autoincrement=False, nullable=True))
    op.add_column('portfolio_info', sa.Column('protect_portfolio', sa.VARCHAR(length=255), autoincrement=False, nullable=True))
    op.drop_column('portfolio_info', 'win_philosophy')
    op.drop_column('portfolio_info', 'unknown_philosophy')
    op.drop_column('portfolio_info', 'monitor_philosophy')
    op.drop_column('portfolio_info', 'lose_philosophy')
    op.drop_column('portfolio_info', 'job_philosophy')
    op.drop_column('portfolio_info', 'games_philosophy')
    op.drop_column('portfolio_info', 'cash')
    # ### end Alembic commands ###

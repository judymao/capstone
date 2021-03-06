"""empty message

Revision ID: 2a6dcf8b0126
Revises: ca8f4a480b13
Create Date: 2020-11-23 22:07:24.677638

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2a6dcf8b0126'
down_revision = 'ca8f4a480b13'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
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
    # ### end Alembic commands ###

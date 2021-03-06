"""empty message

Revision ID: ca8f4a480b13
Revises: 
Create Date: 2020-11-17 12:12:25.877308

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'ca8f4a480b13'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('portfolios')
    op.drop_index('ix_stocks_index', table_name='stocks')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index('ix_stocks_index', 'stocks', ['id'], unique=False)
    op.create_table('portfolios',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('protect_portfolio', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('inv_philosophy', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('next_expenditure', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('name', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('time_horizon', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('holding_constraint', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('trade_size_constraint', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('date', sa.DATE(), autoincrement=False, nullable=True),
    sa.Column('assets', postgresql.ARRAY(sa.VARCHAR(length=255)), autoincrement=False, nullable=True),
    sa.Column('weights', postgresql.ARRAY(postgresql.DOUBLE_PRECISION(precision=53)), autoincrement=False, nullable=True),
    sa.Column('value', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='portfolios_user_id_fkey'),
    sa.PrimaryKeyConstraint('id', name='portfolios_pkey')
    )
    # ### end Alembic commands ###

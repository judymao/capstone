import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True


class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = "postgres://zhuewwnmihdrix:a4f88291040bbb06f5f5c39b0b1837c20937a95f03e8995678aaca7f5c1b65a4@ec2-54-165-164-38.compute-1.amazonaws.com:5432/d6lj7j8bu2ofid"

class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = "postgres://ertfxmkvprwcbp:ba70b3e0bbb13d58fe74c6caaa21edbe304a59e174bbf2a4406630859ebb95cf@ec2-54-158-222-248.compute-1.amazonaws.com:5432/dftg0vldsc4ug8"
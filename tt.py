from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from pilotscope.PilotConfig import PostgreSQLConfig


db_port = "5432"  # database server port (default: "5432")
db_user = "postgres"  # database user name (default: "pilotscope")
db_user_pwd = "li6545991360"  # database user password (default: "pilotscope")
pg_bin_path = "/usr/local/pgsql/bin"  # database bin path, i.e. $PG_PATH/bin (default: None)
pg_data_path = "/home/pgsql/data"  # database data path, i.e. $PG_DATA  (default: None)
config = PostgreSQLConfig(db_port=db_port, db_user=db_user, db_user_pwd=db_user_pwd)
config.enable_deep_control_local(pg_bin_path, pg_data_path)
config.db = 'stats'
config.sql_execution_timeout = 60 * 15
data_interactor = PilotDataInteractor(config)


if __name__ == '__main__':
    q = 'select 1'
    data_interactor.pull_execution_time()
    r = data_interactor.execute(q)
    print(r)
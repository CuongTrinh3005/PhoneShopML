import pandas as pd
import pyodbc

# Config pandas dataframe
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class DBConnector:
    def __init__(self, server_name, username, password, db_name, connection_str):
        self.sever_name = server_name
        self.username = username
        self.password = password
        self.db_name = db_name
        self.connection_str = connection_str
        self.connection = pyodbc.connect(self.connection_str)

    def test_connection(self):
        try:
            db = self.connection
            cursor = db.cursor()
            cursor.execute("SELECT @@VERSION")
            results = cursor.fetchone()
            # Check if anything at all is returned
            if results:
                return True
            else:
                return False
        except pyodbc.Error as ex:
            print(ex)
            print("Error in connection!")
            sqlstate = ex.args[0]
            if sqlstate == '28000':
                print("LDAP Connection failed: check password")
        return False

    def query(self, query_str):
        """This function returns a dataframe of result list"""
        return pd.read_sql_query(query_str, self.connection)

    def all_columns_name(self, table_name):
        """This function returns a dataframe of result list"""
        query_str = f"select COLUMN_NAME from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME='{table_name}'"
        print("All columns names: ", query_str)
        return pd.read_sql_query(query_str, self.connection)

# DBConnector.test_connection = classmethod(DBConnector.test_connection)
# connection_str = "Driver={SQL Server};Server={server_name};UID={username};PWD={password};Database={db_name};"
driver="SQL Server"
servername = 'QUOC-CUONG'
username = 'sa'
password = 'cuong300599'
db_name = 'OnlinePhoneShopJoin'
# str_for_connection = "Driver={SQL Server};Server=QUOC-CUONG;UID=sa;PWD=cuong300599;Database=OnlinePhoneShopJoin;"
str_for_connection = "Driver={driver};Server={servername};UID={username};PWD={password};Database={db_name};"\
    .format(driver=driver, servername=servername, username=username, password=password, db_name=db_name)

connector = DBConnector(servername, username, password, db_name, str_for_connection)
connect_success = connector.test_connection()
if (connect_success):
    print('Connect successfully!')
else:
    print('Connect failed!')

# Get all phones
# query_str = "SELECT * FROM dbo.products pr ,phones p WHERE pr.product_id = p.phone_id "
# df_phones = connector.query(query_str)
# df_phones.head()
# print('All phones are:\n', df_phones)
#
# # Get all accessories
# query_str = "SELECT * FROM dbo.products pr ,accessories a WHERE pr.product_id = a.accessory_id "
# df_accessories = connector.query(query_str)
# df_accessories.head()
# print('All accessories are:\n', df_accessories)

# # Get all column's name
# table_name="phones"
# df_all_columns_names = connector.all_columns_name(table_name)
# print("List of column names of table ", table_name, ":\n", df_all_columns_names)

# # Get all products from a SQL views
# query_str = "SELECT * FROM dbo.all_products"
# df_all_products = connector.query(query_str)
# print("All product list:\n", df_all_products.head())
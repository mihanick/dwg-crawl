using System.Data.SqlServerCe;
using System.IO;

namespace Crawl
{
    class SqlDb
    {
        private string dbPath ;
        private string dataProvider ;
        private SqlCeConnection _conn ;


        public SqlDb()
        {
            dbPath = @"c:\temp\crawl.sdf";
            dataProvider = @"Data Source= "+dbPath;

            CreateTables();

            _conn = new SqlCeConnection(dataProvider);
            _conn.Open();
        }


        public ~SqlDb()
        {
            _conn.Close();
        }

        public void CreateTables()
        {
            //http://stackoverflow.com/questions/6196274/create-sqlce-database-programatically


            // check if exists */
            if (File.Exists(dbPath))
                File.Delete(dbPath);

            // create Database */
            SqlCeEngine engine = new SqlCeEngine(dataProvider);
            engine.CreateDatabase();

            SqlCeConnection conn = new SqlCeConnection(dataProvider);
            conn.Open();


            try
            {
                string createTableSQL = @"CREATE TABLE Data (ObjectId NVARCHAR(256), docHash int, Json NVARCHAR(256), ClassName NVARCHAR(256))";
                SqlCeCommand cmd = new SqlCeCommand(createTableSQL, conn);
                cmd.ExecuteNonQuery();

                createTableSQL = @"CREATE TABLE Files (docHash int UNIQUE, FilePath NVARCHAR(256), docJson NVARCHAR(256))";
                SqlCeCommand cmd2 = new SqlCeCommand(createTableSQL, conn);
                cmd2.ExecuteNonQuery();
            }
            catch (System.Exception ex)
            {
                cDebug.WriteLine(ex.ToString());
            }

            conn.Close();
        }

        public void InsertIntoFiles(string FilePath, int docHash, string docJson)
        {
            string sql = @"INSERT INTO Files (FilePath, docHash, docJson) VALUES (@FilePath, @docHash, @docJson)";
            SqlCeConnection _conn = new SqlCeConnection(dataProvider);

            try
            {
                if (_conn.State != System.Data.ConnectionState.Open)
                    _conn.Open();

                SqlCeCommand command = new SqlCeCommand(sql, _conn);

                command.Parameters.AddWithValue("@FilePath", FilePath);
                command.Parameters.AddWithValue("@docHash", docHash);
                command.Parameters.AddWithValue("@docJson", docJson);

                command.ExecuteNonQuery();

            }
            catch 
            {
            
            }
            finally
            {
                _conn.Close();
            }
        }

        public void SaveObjectData(string objectId, int docHash, string objJson, string objectClassName)
        {
            try
            {
                if (_conn.State != System.Data.ConnectionState.Open)
                    _conn.Open();


                string sql = @"INSERT INTO Data (ObjectId, docHash, Json, ClassName) VALUES (@ObjectId, @docHash, @Json, @ClassName)";
                SqlCeCommand command = new SqlCeCommand(sql, _conn);

                command.Parameters.Add("@ObjectId", objectId);
                command.Parameters.Add("@docHash", docHash);
                command.Parameters.Add("@ClassName", objectClassName);

                command.Parameters.Add("@Json", objJson);

                command.ExecuteNonQuery();

            }
            catch (System.Exception e)
            {
                cDebug.WriteLine("Ошибка сохранения объекта {0}: {1}", objectClassName, e.Message);
            }
            finally
            {
                _conn.Close();
            }
        }

        
    }
}
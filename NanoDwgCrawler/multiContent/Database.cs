using System.Data.SqlServerCe;
using System.IO;

namespace Crawl
{
    class SqlDb
    {
        private string dbPath;
        private string dataProvider;
        private SqlCeConnection _conn;


        public SqlDb()
        {
            dbPath = @"C:\Users\Mike Gladkikh\Documents\GitHub\dwg-crawl\NanoDwgCrawler\crawl.sdf";
            dataProvider = @"Data Source= " + dbPath;

            CreateTables();

            _conn = new SqlCeConnection(dataProvider);
            _conn.Open();
        }


        ~SqlDb()
        {
            _conn.Close();
        }

        void CreateTables()
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


            string createTableSQL = @"CREATE TABLE Data (ObjectId NVARCHAR(256), Json NTEXT, ClassName NVARCHAR(256), FilePath NVARCHAR(4000))";
            SqlCeCommand cmd = new SqlCeCommand(createTableSQL, conn);
            cmd.ExecuteNonQuery();

            createTableSQL = @"CREATE TABLE Files (FilePath NVARCHAR(4000), docJson NTEXT)";
            SqlCeCommand cmd2 = new SqlCeCommand(createTableSQL, conn);
            cmd2.ExecuteNonQuery();

            conn.Close();
        }

        public void InsertIntoFiles(string FilePath, string docJson)
        {
            string sql = @"INSERT INTO Files (FilePath, docJson) VALUES (@FilePath, @docJson)";

            if (_conn.State == System.Data.ConnectionState.Open)
            {
                SqlCeCommand command = new SqlCeCommand(sql, _conn);

                command.Parameters.AddWithValue("@FilePath", FilePath);
                command.Parameters.AddWithValue("@docJson", docJson);

                command.ExecuteNonQuery();
            }
        }

        public void SaveObjectData(string objectId, string objJson, string objectClassName, string filePath)
        {
            if (_conn.State == System.Data.ConnectionState.Open)
            {
                string sql = @"INSERT INTO Data (ObjectId, Json, ClassName, FilePath) VALUES (@ObjectId, @Json, @ClassName, @FilePath)";
                SqlCeCommand command = new SqlCeCommand(sql, _conn);

                command.Parameters.Add("@ObjectId", objectId);
                command.Parameters.Add("@ClassName", objectClassName);
                command.Parameters.Add("@Json", objJson);
                command.Parameters.Add("@FilePath", filePath);

                command.ExecuteNonQuery();
            }
        }
    }
}
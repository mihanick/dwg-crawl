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


            string createTableSQL = @"CREATE TABLE Data (ObjectId NVARCHAR(256), Json NTEXT, ClassName NVARCHAR(256), FileId NVARCHAR(256))";
            SqlCeCommand cmd = new SqlCeCommand(createTableSQL, conn);
            cmd.ExecuteNonQuery();

            createTableSQL = @"CREATE TABLE Files (FilePath NVARCHAR(4000), docJson NTEXT, FileId NVARCHAR(256))";
            SqlCeCommand cmd2 = new SqlCeCommand(createTableSQL, conn);
            cmd2.ExecuteNonQuery();

            conn.Close();
        }

        public void InsertIntoFiles(string FilePath, string docJson, string fileId)
        {
            string sql = @"INSERT INTO Files (FilePath, docJson, FileId) VALUES (@FilePath, @docJson, @FileId)";

            if (_conn.State == System.Data.ConnectionState.Open)
            {
                SqlCeCommand command = new SqlCeCommand(sql, _conn);

                command.Parameters.AddWithValue("@FilePath", FilePath);
                command.Parameters.AddWithValue("@docJson", docJson);
                command.Parameters.AddWithValue("@FileId", fileId);

                command.ExecuteNonQuery();
            }
        }

        public void SaveObjectData(string objectId, string objJson, string objectClassName, string fileId)
        {
            if (_conn.State == System.Data.ConnectionState.Open)
            {
                string sql = @"INSERT INTO Data (ObjectId, Json, ClassName, FileId) VALUES (@ObjectId, @Json, @ClassName, @FileId)";
                SqlCeCommand command = new SqlCeCommand(sql, _conn);

                command.Parameters.Add("@ObjectId", objectId);
                command.Parameters.Add("@ClassName", objectClassName);
                command.Parameters.Add("@Json", objJson);
                command.Parameters.Add("@FileId", fileId);

                command.ExecuteNonQuery();
            }
        }
    }
}
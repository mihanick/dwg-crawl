using System.Data.SqlServerCe;
using System.IO;
using System.Collections.Generic;

namespace Crawl
{
    public class SqlDb
    {
        private string dbPath;
        private string dataProvider;
        private SqlCeConnection _conn;

        public SqlDb()
        {
            dbPath = @"F:\Data\crawl3.sdf";
            dataProvider = @"Data Source= " + dbPath + "; Max Database Size=4091";

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
            if (!File.Exists(dbPath))
            {
                // create Database */
                SqlCeEngine engine = new SqlCeEngine(dataProvider);
                engine.CreateDatabase();
            }

            SqlCeConnection conn = new SqlCeConnection(dataProvider);
            conn.Open();

            if (!TableExist("Data", conn))
            {
                string createTableSQL = @"CREATE TABLE Data (ObjectId NVARCHAR(256), Json NTEXT, ClassName NVARCHAR(256), FileId NVARCHAR(256))";
                SqlCeCommand cmd = new SqlCeCommand(createTableSQL, conn);
                cmd.ExecuteNonQuery();
            }

            if (!TableExist("Files", conn))
            {
                string createTableSQL = @"CREATE TABLE Files (FilePath NVARCHAR(4000), docJson NTEXT, FileId NVARCHAR(256), FileHash NVARCHAR(256), Scanned BIT NOT NULL)";
                SqlCeCommand cmd = new SqlCeCommand(createTableSQL, conn);
                cmd.ExecuteNonQuery();
            }

            conn.Close();
        }


        public void InsertIntoFiles(string FilePath, string docJson, string fileId, string fileHash)
        {
            if (_conn.State == System.Data.ConnectionState.Open)
            {
                //http://stackoverflow.com/questions/12219324/how-to-check-value-exists-on-sql-table
                SqlCeCommand chkHashExist = new SqlCeCommand("SELECT FileHash FROM Files WHERE (FileHash='" + fileHash + "' AND FilePath='"+FilePath+"')", _conn);

                object qryRslt = chkHashExist.ExecuteScalar();
                string sRslt = (string)qryRslt;

                if (qryRslt == null)
                {
                    try
                    {
                        string sql = @"INSERT INTO Files (FilePath, docJson, FileId, FileHash, Scanned) VALUES (@FilePath, @docJson, @FileId, @FileHash,@Scanned)";

                        SqlCeCommand command = new SqlCeCommand(sql, _conn);

                        command.Parameters.AddWithValue("@FilePath", FilePath);
                        command.Parameters.AddWithValue("@docJson", docJson);
                        command.Parameters.AddWithValue("@FileId", fileId);
                        command.Parameters.AddWithValue("@FileHash", fileHash);
                        command.Parameters.AddWithValue("@Scanned", 0);

                        command.ExecuteNonQuery();
                    }
                    catch
                    {
                    }
                }
            }
        }

        public void SaveObjectData(string objectId, string objJson, string objectClassName, string fileId)
        {
            if (_conn.State == System.Data.ConnectionState.Open)
            {
                string sql = @"INSERT INTO Data (ObjectId, Json, ClassName, FileId) VALUES (@ObjectId, @Json, @ClassName, @FileId)";
                SqlCeCommand command = new SqlCeCommand(sql, _conn);
                try
                {
                    command.Parameters.Add("@ObjectId", objectId);
                    command.Parameters.Add("@ClassName", objectClassName);
                    command.Parameters.Add("@Json", objJson);
                    command.Parameters.Add("@FileId", fileId);

                    command.ExecuteNonQuery();
                }
                catch(System.Exception e)
                {
                    cDebug.WriteLine("Save to db failed: " + e.Message);
                }
            }
        }

        private bool TableExist(string TableName, SqlCeConnection connection)
        {

            string commandTxt = "SELECT COUNT(*) FROM Information_Schema.Tables WHERE (TABLE_NAME = '" + TableName + "')";
            SqlCeCommand command = new SqlCeCommand(commandTxt, connection);
            int result = (int)command.ExecuteScalar();
            return result != 0;
        }

        internal CrawlDocument GetNewRandomUnscannedDocument()
        {
            CrawlDocument result = null;
            if (_conn.State != System.Data.ConnectionState.Open)
                return result;

            //Check db size is close to maximum
            FileInfo Fi = new FileInfo(dbPath);
            long maxsize = 2000*1024*1024;
            if (Fi.Length > maxsize)
                return null;


            //http://stackoverflow.com/questions/13665309/how-to-randomly-select-one-row-off-table-based-on-critera
            //https://msdn.microsoft.com/en-us/library/cc441928.aspx

            string commandTxt = 
                @"SELECT        FilePath, docJson, FileId, FileHash, Scanned
                FROM            Files
                WHERE        (Scanned = '0')
                ORDER BY NEWID()";

            SqlCeCommand command = new SqlCeCommand(commandTxt, _conn);
            SqlCeDataReader dr = command.ExecuteReader();

            while (dr.Read())
            {
                //http://stackoverflow.com/questions/4018114/read-data-from-sqldatareader
                result = new CrawlDocument();
                result.FileId = dr["FileId"].ToString();
                result.Hash = dr["FileHash"].ToString();
                result.Path = dr["FilePath"].ToString();
                break;
            }
            return result;
        }

        public void SetDocumentScanned(string fileId)
        {
            if (_conn.State == System.Data.ConnectionState.Open)
            {
                string sql = @"UPDATE Files SET Scanned=1 WHERE (FileId=@FileId)";
                SqlCeCommand command = new SqlCeCommand(sql, _conn);

                command.Parameters.Add("@FileId", fileId);

                command.ExecuteNonQuery();
            }
        }

        public List<string> GetObjectJsonByClassName(string className)
        {
            List<string> result = new List<string>();

            if (_conn.State != System.Data.ConnectionState.Open)
                return result;


            //http://stackoverflow.com/questions/13665309/how-to-randomly-select-one-row-off-table-based-on-critera
            //https://msdn.microsoft.com/en-us/library/cc441928.aspx

            string commandTxt =
                @"SELECT        Json, ClassName
                FROM            Data
                WHERE        (ClassName = '"+className+"')";

            SqlCeCommand command = new SqlCeCommand(commandTxt, _conn);
            SqlCeDataReader dr = command.ExecuteReader();

            while (dr.Read())
            {
                //http://stackoverflow.com/questions/4018114/read-data-from-sqldatareader

                string json =  dr["Json"].ToString();
                result.Add(json);

            }
            return result;
        }
    }
}
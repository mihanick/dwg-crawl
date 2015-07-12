namespace Crawl
{
    using MongoDB;
    using MongoDB.Bson;
    using MongoDB.Driver;
    using System;
    using System.Collections.Generic;

    public class DbMongo
    {
        private string DbName;
        private MongoClient clientMongo;
        private MongoDatabase databaseMongo;

        public DbMongo(string dbName="")
        {
            if (dbName == "")
                DbName = "geometry";
            else
                DbName = dbName;

            clientMongo = new MongoClient();
            databaseMongo = clientMongo.GetServer().GetDatabase(DbName);

            CreateTables();
       }

        public void Clear()
        {
            clientMongo.GetServer().GetDatabase(DbName).DropCollection("objects");
            clientMongo.GetServer().GetDatabase(DbName).DropCollection("files");            
        }

        void CreateTables()
        {
            // Seed initial data
            if (!databaseMongo.CollectionExists("objects"))
            {
                // http://mongodb.github.io/mongo-csharp-driver/2.0/getting_started/quick_tour/
                // Seed with object data
                var collection = databaseMongo.GetCollection<BsonDocument>("objects");

                string docJson  =
                @"{
                'ObjectId': '1',
                'FileId':'7ad00d95-f663-4db9-b379-1ff0f30a616d',
                        'ClassName': 'AcDbLine',
                        'Color': 'BYLAYER',
                        'EndPoint': {
                            'ClassName': 'Point3D',
                            'X': 294742.67173893179,
                            'Y': 93743.0034136844,
                            'Z': 0
                        },
                        'Layer': 'СО_Выноски',
                        'Length': 150,
                        'LineWeight': 'LineWeight040',
                        'Linetype': 'Continuous',
                        'StartPoint': {
                            'ClassName': 'Point3D',
                            'X': 294742.67173893179,
                            'Y': 93893.0034136844,
                            'Z': 0
                        }
                    }";

                BsonDocument doc = BsonDocument.Parse(docJson);

                collection.Insert(doc);
            }

            if (!databaseMongo.CollectionExists("files"))
            {
                var collection = databaseMongo.GetCollection<BsonDocument>("files");
                string docJson =
                    @"{
                        'FileId':'7ad00d95-f663-4db9-b379-1ff0f30a616d',
                        'Hash':'a4733bbed995e26a389c5489402b3cee',
                        'Path':'D:\\Documents\\Dropbox\\CrawlDwgs\\084cdbd1-cb5f-4380-a04a-f577d85a7bbb.dwg',
                        'Scanned':false
                    }";
                BsonDocument doc = BsonDocument.Parse(docJson);

                collection.Insert(doc);
            }
        }

        public void InsertIntoFiles(string FilePath, string docJson, string fileId, string fileHash)
        {

            // Check hash alreade exists, if no - insert
            // if yes - skip

        }

        public void SaveObjectData(string objectId, string objJson, string objectClassName, string fileId)
        {
            // Just save
        }



        internal CrawlDocument GetNewRandomUnscannedDocument()
        {
            CrawlDocument result = null;
            /*
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
             */
            return result;
        }

        public void SetDocumentScanned(string fileId)
        {
            /*
            if (_conn.State == System.Data.ConnectionState.Open)
            {
                string sql = @"UPDATE Files SET Scanned=1 WHERE (FileId=@FileId)";
                SqlCeCommand command = new SqlCeCommand(sql, _conn);

                command.Parameters.Add("@FileId", fileId);

                command.ExecuteNonQuery();
            }
             */
        }

        public List<string> GetObjectJsonByClassName(string className)
        {
            List<string> result = new List<string>();

            QueryDocument filter = new QueryDocument("ClassName", className);
            var objJsons = databaseMongo.GetCollection("objects").Find(filter);

            foreach (var anObject in objJsons)
                result.Add(anObject.ToString());

            return result;
        }
    }
}
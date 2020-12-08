using MongoDB.Bson;
using MongoDB.Driver;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace DwgDump.Db
{
	public class DbMongo
	{
		private readonly string dbName;
		private MongoClient client;
		private IMongoDatabase db;

		private IMongoCollection<BsonDocument> files;
		private IMongoCollection<BsonDocument> objects;

		public const string DefaultDatabaseName = "geometry";

		public DbMongo()
		{
			this.dbName = "geometry";
			this.Create();
		}

		public DbMongo(string dbName)
		{
			if (string.IsNullOrEmpty(dbName))
				this.dbName = DefaultDatabaseName;
			else
				this.dbName = dbName;

			Create();
		}

		// StackOverflow
		private bool CollectionExists(
			IMongoDatabase database, string collectionName)
		{
			// https://stackoverflow.com/questions/25017219/how-to-check-if-collection-exists-in-mongodb-using-c-sharp-driver
			var filter = new BsonDocument("name", collectionName);
			var options = new ListCollectionNamesOptions { Filter = filter };

			return database.ListCollectionNames(options).Any();
		}

		private void Create()
		{
			client = new MongoClient();

			db = client.GetDatabase(this.dbName);

			// files = DatabaseMongo.GetCollection<BsonDocument>("files");

			if (!CollectionExists(db, "files"))
				db.CreateCollection("files");

			files = db.GetCollection<BsonDocument>("files");

			//files.CreateIndex("FileId");
			//files.CreateIndex("BlockId");
			//files.CreateIndex("ClassName");

			if (!CollectionExists(db, "objects"))
				db.CreateCollection("objects");

			objects = db.GetCollection<BsonDocument>("objects");

			//objects.CreateIndex("ClassName");
			//objects.CreateIndex("ObjectId");
			//objects.CreateIndex("FileId");
		}

		public void Clear()
		{
			client.GetDatabase(dbName).DropCollection("objects");
			client.GetDatabase(dbName).DropCollection("files");
		}

		public void InsertIntoFiles(string docJson)
		{
			BsonDocument doc = BsonDocument.Parse(docJson);
			bool docIsAFile = doc["ClassName"].ToString() == "File";

			if (docIsAFile)
			{
				string hash = doc["Hash"].ToString();
				doc["Scanned"] = false;

				var filter = new QueryDocument
				{
					{ "Hash", hash },
					{ "ClassName", "File" }
				};

				var qryResult = files.Find(filter);
				// if hash exist - we should skip insertion
				if (qryResult.CountDocuments() == 0)
					// Check hash already exists, if no - insert
					files.InsertOne(doc);
			}
			else
			{
				doc["Scanned"] = true;
				files.InsertOne(doc);
			}
		}

		public void InsertIntoFiles(CrawlDocument crawlDocument)
		{
			BsonDocument doc = crawlDocument.ToBsonDocument();

			var filter = new QueryDocument("Hash", crawlDocument.Hash);
			
			if (files.Find(filter).CountDocuments() == 0)
				// Check hash already exists, if no - insert
				files.InsertOne(doc);
		}

		public void SaveObjectData(string objJson, string fileId)
		{
			try
			{
				if (string.IsNullOrEmpty(objJson))
					return;

				BsonDocument doc = BsonDocument.Parse(objJson);
				doc["FileId"] = fileId;
				//doc.Add("FileId", fileId);
				objects.InsertOne(doc);
			}
			catch (System.Exception e)
			{
				Debug.WriteLine(e.Message);
			}
		}

		public CrawlDocument GetNewRandomUnscannedDocument()
		{
			QueryDocument filter = new QueryDocument
			{
				{ "Scanned", false },
				{ "ClassName", "File" }
			};

			// Not efficient to obtain all collection, but 'files' cooolection shouldn't bee too large
			// http://stackoverflow.com/questions/3975290/produce-a-random-number-in-a-range-using-c-sharp
			// Get random document
			Random r = new Random((int)DateTime.Now.Ticks);
			long num = files.CountDocuments(filter);//Max range
			int x = r.Next((int)num);
			var aFile = files.Find<BsonDocument>(filter).Skip(x).Limit(1);
			if (aFile.CountDocuments() == 0)
				return null;

			var file = aFile.FirstOrDefault().ToBsonDocument();
			CrawlDocument result = new CrawlDocument
			{
				ClassName = "File",
				FileId = file["FileId"].ToString(),
				Hash = file["Hash"].ToString(),
				Path = file["Path"].ToString(),
				Scanned = file["Scanned"].ToBoolean()
			};

			// Check db size is close to maximum
			// FileInfo Fi = new FileInfo(dbPath);
			// long maxsize = 2000*1024*1024;
			// if (Fi.Length > maxsize)
			// return null;

			return result;
		}

		public List<CrawlDocument> GetFile(string fileId)
		{
			QueryDocument filter = new QueryDocument
			{
				{ "FileId", fileId },
				{ "ClassName", "File" }
			};


			var allFiles = files.Find(filter).ToEnumerable();

			var result = files.Find(filter).ToEnumerable()
				.Select(doc =>
					new CrawlDocument()
					{
						FileId = doc["FileId"].ToString(),
						Hash = doc["Hash"].ToString(),
						Path = doc["Path"].ToString(),
						Scanned = doc["Scanned"].ToBoolean()
					});

			return result.ToList();
		}

		public void SetDocumentScanned(string fileId)
		{
			var filter = new QueryDocument("FileId", fileId);
			var update = Builders<BsonDocument>.Update.Set("Scanned", true);
			files.UpdateOne(filter, update);
		}

		public bool HasFileId(string FileId)
		{
			QueryDocument filter = new QueryDocument("FileId", FileId);
			
			return files.Find(filter).CountDocuments() > 0;
		}

		public bool HasFileHash(string FileHash)
		{
			QueryDocument filter = new QueryDocument("Hash", FileHash);
			
			return files.Find(filter).CountDocuments() > 0;
		}

		public bool HasObject(string objectId)
		{
			QueryDocument filter = new QueryDocument("ObjectId", objectId);

			return objects.Find(filter).CountDocuments() > 0;
		}
	}
}
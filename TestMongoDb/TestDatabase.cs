namespace Crawl
{
    using System;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Crawl;
    using System.Collections.Generic;
    using System.Diagnostics;

    [TestClass]
    public class TestDatabase
    {
        DbMongo db;

        public TestDatabase()
        {
            db = new DbMongo("testDb");
        }

        [TestMethod]
        public void TestDbCreate()
        {
            db.Clear();
            db.Seed();

            Assert.IsTrue(db.HasFileHash("a4733bbed995e26a389c5489402b3cee"));
            Assert.IsTrue(db.HasFileId("7ad00d95-f663-4db9-b379-1ff0f30a616d"));
            Assert.IsTrue(db.HasObject("12345678"));
        }

        [TestMethod]
        public void TestInsertIntoFiles()
        {
            db.Clear();
            string json1 = @"
            {
	            'ClassName': 'File',
	            'FileId': 'bc6a1669-51ce-444c-94c6-cfec71c0f44d',
	            'Hash': 'd520b80512f226e81dd72294037657fd',
	            'Path': '\\\\FILESERVER\\home\\#АРХИВ 2014\\Объекты\\МНОГОТОПЛИВНАЯ АЗС №15\\задание на фундаменты.dwg',
	            'Scanned': false,
	            '_id': {
		            '$oid': '55a49dfff80dc7180c8228d3'
	            }
            }";
            string json2 = @"
            {
	            'ClassName': 'File',
	            'FileId': '9e2769ff-678f-401b-8d10-e0581aa6bf98',
	            'Hash': '253ffb6063333c5bfc1109c5d7db1945',
	            'Path': '\\\\FILESERVER\\home\\#АРХИВ 2014\\Объекты\\МНОГОТОПЛИВНАЯ АЗС №15\\образец исх данные.dwg',
	            'Scanned': false,
	            '_id': {
		            '$oid': '55a49dfff80dc7180c8228d4'
	            }
            }
            ";
            db.InsertIntoFiles(json1);
            db.InsertIntoFiles(json2);

            Assert.IsTrue(db.HasFileHash("d520b80512f226e81dd72294037657fd"));
            Assert.IsTrue(db.HasFileId("bc6a1669-51ce-444c-94c6-cfec71c0f44d"));
            Assert.IsTrue(db.HasFileHash("253ffb6063333c5bfc1109c5d7db1945"));
            Assert.IsTrue(db.HasFileId("9e2769ff-678f-401b-8d10-e0581aa6bf98"));

            db.Clear();

            Crawl.CrawlDocument cdoc = new CrawlDocument(@"D:\Documents\Desktop\SingleFile\+b3826065-07d1-4d4a-8af4-35ebc3630117.dwg");

            db.InsertIntoFiles(cdoc);

            Assert.IsTrue(db.HasFileHash(cdoc.Hash));
            Assert.IsTrue(db.HasFileId(cdoc.FileId));
        }

        [TestMethod]
        public void TestSaveObjectData()
        {
            db.Clear();

            string json = @"{
                'ObjectId': '12345678',
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

            string fileId = "7ad00d95-f663-4db9-b379-1ff0f30a616d";

            db.SaveObjectData(json, fileId);

            Assert.IsTrue(db.HasObject("12345678"));
        }

        [TestMethod]
        public void TestGetObjectJsonByClassName()
        {
            db.Seed();

            List<string> result = db.GetObjectJsonByClassName("AcDbLine");

            Assert.IsTrue(result.Count > 0);
            Assert.IsTrue(result[0].Contains("AcDbLine"));
        }

        [TestMethod]
        public void TestSetDocumentScanned()
        {
            db.Clear();

            string expectedFileId = "bc6a1669-51ce-444c-94c6-cfec71c0f44d";
            string jsonFile = @"
            {
	            'ClassName': 'File',
	            'FileId': 'bc6a1669-51ce-444c-94c6-cfec71c0f44d',
	            'Hash': 'd520b80512f226e81dd72294037657fd',
	            'Path': '\\\\FILESERVER\\home\\#АРХИВ 2014\\Объекты\\МНОГОТОПЛИВНАЯ АЗС №15\\задание на фундаменты.dwg',
	            'Scanned': false
            }";
            string jsonPxy = @"
            {
	            'ClassName': 'AcDbProxy',
	            'FileId': 'bc6a1669-51ce-444c-94c6-cfec71c0f44d',
	            'Hash': 'd520b80512f226e81dd72294037657fd',
	            'Path': '\\\\FILESERVER\\home\\#АРХИВ 2014\\Объекты\\МНОГОТОПЛИВНАЯ АЗС №15\\задание на фундаменты.dwg',
	            'Scanned': false
            }";
            string jsonBlk = @"
            {
	            'ClassName': 'AcDbBlockTableReference',
	            'FileId': 'bc6a1669-51ce-444c-94c6-cfec71c0f44d',
	            'Hash': 'd520b80512f226e81dd72294037657fd',
	            'Path': '\\\\FILESERVER\\home\\#АРХИВ 2014\\Объекты\\МНОГОТОПЛИВНАЯ АЗС №15\\задание на фундаменты.dwg',
	            'Scanned': false
            }";

            db.InsertIntoFiles(jsonFile);
            db.InsertIntoFiles(jsonPxy);
            db.InsertIntoFiles(jsonBlk);

            db.SetDocumentScanned(expectedFileId);
            List<CrawlDocument> docList = db.GetFile(expectedFileId);
            Assert.AreEqual(1, docList.Count);

            foreach (CrawlDocument cd in docList)
            {
                Assert.AreEqual(expectedFileId, cd.FileId);
                Assert.IsTrue(cd.Scanned);
            }
        }

        [TestMethod]
        public void TestGetNewRandomUnscannedDocument()
        {
            db.Clear();
            int numRecords = 1000;

            Stopwatch timer = new Stopwatch();
            for (int i = 0; i < numRecords; i++)
            {
                CrawlDocument cd = new CrawlDocument();
                cd.FileId = Guid.NewGuid().ToString();
                cd.Hash = Guid.NewGuid().ToString();
                cd.Path = i.ToString();
                cd.Scanned = false;
                db.InsertIntoFiles(cd);
            }

            CrawlDocument cd1 = db.GetNewRandomUnscannedDocument();

            CrawlDocument cd2 = db.GetNewRandomUnscannedDocument();

            // Random-selected files from 1000 records should differ
            Assert.IsFalse(cd1.FileId == cd2.FileId);

            timer.Stop();
            // Assume 3ms for each record should be enough
            Assert.IsTrue(timer.ElapsedMilliseconds < 3*numRecords);
        }
    }
}

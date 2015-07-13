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
            string json = @"{
                        'FileId':'7ad00d95-f663-4db9-b379-1ff0f30a616d',
                        'Hash':'a4733bbed995e26a389c5489402b3cee',
                        'Path':'D:\\Documents\\Dropbox\\CrawlDwgs\\084cdbd1-cb5f-4380-a04a-f577d85a7bbb.dwg',
                    }";

            db.InsertIntoFiles("D:\\Documents\\Dropbox\\CrawlDwgs\\084cdbd1-cb5f-4380-a04a-f577d85a7bbb.dwg", json, "7ad00d95-f663-4db9-b379-1ff0f30a616d", "a4733bbed995e26a389c5489402b3cee");

            Assert.IsTrue(db.HasFileHash("a4733bbed995e26a389c5489402b3cee"));
            Assert.IsTrue(db.HasFileId("7ad00d95-f663-4db9-b379-1ff0f30a616d"));

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
            db.Seed();
            string json = @"{
                        'FileId':'7ad00d95-f663-4db9-b379-1ff0f30a616d',
                        'Hash':'a4733bbed995e26a389c5489402b3cee',
                        'Path':'D:\\Documents\\Dropbox\\CrawlDwgs\\084cdbd1-cb5f-4380-a04a-f577d85a7bbb.dwg',
                    }";

            db.InsertIntoFiles("AcDbProxy",
                json, "7ad00d95-f663-4db9-b379-1ff0f30a616d", "a4733bbed995e26a389c5489402b3cee");

            db.InsertIntoFiles("AcDbBlockTableReference",
                json, "7ad00d95-f663-4db9-b379-1ff0f30a616d", "a4733bbed995e26a389c5489402b3cee");

            db.SetDocumentScanned("7ad00d95-f663-4db9-b379-1ff0f30a616d");

            foreach (CrawlDocument cd in db.GetFile("7ad00d95-f663-4db9-b379-1ff0f30a616d"))
                Assert.IsTrue(cd.Scanned);
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
            Assert.IsTrue(timer.ElapsedMilliseconds < numRecords);
        }
    }
}

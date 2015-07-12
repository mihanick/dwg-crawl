namespace Crawl
{
    using System;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Crawl;

    [TestClass]
    public class TestDatabase
    {
        [TestMethod]
        public void TestDbExists()
        {
            var db = new DbMongo("testDb");

            Assert.IsNotNull(db);

            db.Clear();
        }
    }
}

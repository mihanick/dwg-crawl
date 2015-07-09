namespace Crawl.Test
{
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;

    [TestClass]
    public class TestRectangleTree
    {
        RectangleTree rt;
        Rectangle[] rectangles;

        public TestRectangleTree()
        {
            rt = new RectangleTree();
            rectangles = new Rectangle[7];

            rectangles[0] = new Rectangle(new crawlPoint3d(0, 0, 0), new crawlPoint3d(100, 100, 0));
            rectangles[1] = new Rectangle(new crawlPoint3d(50, 50, 0), new crawlPoint3d(350, 150, 0));
            rectangles[2] = new Rectangle(new crawlPoint3d(550, 250, 0), new crawlPoint3d(650, 350, 0));
            rectangles[3] = new Rectangle(new crawlPoint3d(575, 275, 0), new crawlPoint3d(625, 325, 0));
            rectangles[4] = new Rectangle(new crawlPoint3d(750, 50, 0), new crawlPoint3d(850, 150, 0));
            rectangles[5] = new Rectangle(new crawlPoint3d(750, 150, 0), new crawlPoint3d(850, 250, 0));
            rectangles[6] = new Rectangle(new crawlPoint3d(800, 100, 0), new crawlPoint3d(900, 200, 0));
        }

        [TestMethod]
        public void TestAddTime()
        {
            Stopwatch timer = Stopwatch.StartNew();

            for (int i = 0; i < rectangles.Length; i++)
                rt.Add(rectangles[i]);

            timer.Stop();
            Assert.IsTrue(timer.ElapsedMilliseconds < rectangles.Length);
        }

        [TestMethod]
        public void TestSearchIntersectionsTime()
        {
            Rectangle searchedArea = new Rectangle(new crawlPoint3d(0, 0, 0), new crawlPoint3d(350, 150, 0));

            for (int i = 0; i < rectangles.Length; i++)
                rt.Add(rectangles[i]);

            Stopwatch timer = Stopwatch.StartNew();
            List<Rectangle> result = rt.Intersections(searchedArea);
            timer.Stop();
            Assert.IsTrue(timer.ElapsedMilliseconds < 1);
        }

        [TestMethod]
        public void TestSearchIntersectionResultSimple()
        {
            Rectangle searchedArea = new Rectangle(new crawlPoint3d(0, 0, 0), new crawlPoint3d(350, 150, 0));

            for (int i = 0; i < rectangles.Length; i++)
                rt.Add(rectangles[i]);

            List<Rectangle> result = rt.Intersections(searchedArea);

            Assert.AreEqual(2, result.Count);

            CollectionAssert.Contains(result, rectangles[0]);
            CollectionAssert.Contains(result, rectangles[1]);
        }

        [TestMethod]
        public void TestIntersections()
        {
            for (int i = 0; i < rectangles.Length; i++)
                rt.Add(rectangles[i]);

            List<Rectangle> result = rt.Intersections(rectangles[0]);

            Assert.AreEqual(2, result.Count);

            Assert.IsTrue(result[0].Equals(rectangles[0]));
            Assert.IsTrue(result[1].Equals(rectangles[1]));

            result = rt.Intersections(rectangles[4]);

            Assert.AreEqual(3, result.Count);

            Assert.IsTrue(result[0].Equals(rectangles[4]));
            Assert.IsTrue(result[1].Equals(rectangles[5]));
            Assert.IsTrue(result[2].Equals(rectangles[6]));
        }

        [TestMethod]
        public void TestInclusions()
        {
            for (int i = 0; i < rectangles.Length; i++)
                rt.Add(rectangles[i]);

            List<Rectangle> result = rt.Inclusions(rectangles[2]);

            Assert.AreEqual(1, result.Count);

            Assert.IsTrue(result[0].Equals(rectangles[3]));
        }

        [TestMethod]
        public void TestRectangleTreeAllContents()
        {
            for (int i = 0; i < rectangles.Length; i++)
                rt.Add(rectangles[i]);

            List<Rectangle> contents = rt.GetContents();

            Assert.AreEqual(7, contents.Count);

            foreach (Rectangle rec in rectangles)
                Assert.IsTrue(contents.Contains(rec));
        }

        [TestMethod]
        public void TestRectangleTreeStructure()
        {
            for (int i = 0; i < rectangles.Length; i++)
                rt.Add(rectangles[i]);

            string contents = rt.Root.ConvertToString();

            string expected =
@"(0, 0), (900, 350)
	(0, 0), (650, 350)
		(0, 0), (350, 150)
			(0, 0), (100, 100)
				rec:(0, 0), (100, 100)
			(50, 50), (350, 150)
				rec:(50, 50), (350, 150)

		(550, 250), (650, 350)
			rec:(550, 250), (650, 350)
		(575, 275), (625, 325)
			rec:(575, 275), (625, 325)

	(750, 50), (850, 150)
		rec:(750, 50), (850, 150)
	(750, 150), (850, 250)
		rec:(750, 150), (850, 250)
	(800, 100), (900, 200)
		rec:(800, 100), (900, 200)
";

            Assert.AreEqual(expected, contents);
        }

        [TestMethod]
        public void TestSearchResultAndTimeMedium()
        {
            Rectangle searchedArea = new Rectangle(new crawlPoint3d(-24, -34, 0), new crawlPoint3d(401, 74, 0));

            SqlDb sqlDB = new SqlDb(@"C:\Data\rectangle.sdf");
            List<string> jsonOfLines = sqlDB.GetObjectJsonByClassName("AcDbLine");

            foreach (string jsonLine in jsonOfLines)
            {
                crawlAcDbLine line = jsonHelper.From<crawlAcDbLine>(jsonLine);
                // Limiting all junk small lines
                if (line.Length > 10)
                {
                    Rectangle rec = new Rectangle(line.StartPoint, line.EndPoint);
                    rt.Add(rec);
                }
            }

            Stopwatch timer = Stopwatch.StartNew();
            List<Rectangle> result = rt.Inclusions(searchedArea);
            timer.Stop();
            Assert.IsTrue(timer.ElapsedMilliseconds < Math.Log(jsonOfLines.Count));

            Assert.AreEqual(2, result.Count);

            int grade = 0;
            foreach (var rect in result)
            {
                if (rect.pointA.Equals(new crawlPoint3d(0, 0, 0)) && rect.pointC.Equals(new crawlPoint3d(100, 0, 0)))
                    grade++;
                if (rect.pointA.Equals(new crawlPoint3d(50, 50, 0)) && rect.pointC.Equals(new crawlPoint3d(350, 50, 0)))
                    grade++;
            }

            Assert.AreEqual(2, grade);
        }

        [TestMethod]
        public void TestSearchResultAndTimeLarge()
        {
            Rectangle searchedArea = new Rectangle(new crawlPoint3d(115184, 29374, 0), new crawlPoint3d(133962, 35634, 0));

            SqlDb sqlDB = new SqlDb(@"C:\Data\SingleFile.sdf");
            List<string> jsonOfLines = sqlDB.GetObjectJsonByClassName("AcDbLine");

            int numberOfLinesInsideSearchedArea = 0;

            foreach (string jsonLine in jsonOfLines)
            {
                crawlAcDbLine line = jsonHelper.From<crawlAcDbLine>(jsonLine);
                // Limiting all junk small lines
                if (line.Length > 0)
                {
                    Rectangle rec = new Rectangle(line.StartPoint, line.EndPoint);
                    rt.Add(rec);

                    if (searchedArea.Includes(line.StartPoint) && searchedArea.Includes(line.EndPoint))
                        numberOfLinesInsideSearchedArea++;
                }
            }

            Stopwatch timer = Stopwatch.StartNew();
            List<Rectangle> result = rt.Inclusions(searchedArea);
            timer.Stop();
            Assert.IsTrue(timer.ElapsedMilliseconds < Math.Log(jsonOfLines.Count));

            Assert.AreEqual(numberOfLinesInsideSearchedArea, result.Count);

        }

        [TestMethod]
        public void TestCreateBigTree()
        {
            SqlDb sqlDB = new SqlDb(@"C:\Data\crawl2.sdf");
            List<string> jsonOfLines = sqlDB.GetObjectJsonByClassName("AcDbLine");

            List<crawlAcDbLine> lines = new List<crawlAcDbLine>();

            foreach (string jsonLine in jsonOfLines)
            {
                try
                {
                    crawlAcDbLine cLine = jsonHelper.From<crawlAcDbLine>(jsonLine);
                    lines.Add(cLine);
                }
                catch { }
            }

            Stopwatch timer = Stopwatch.StartNew();
            foreach (var line in lines)
            {
                Rectangle rec = new Rectangle(line.StartPoint, line.EndPoint);
                rt.Add(rec);
            }

            timer.Stop();
            Assert.IsTrue(timer.ElapsedMilliseconds < lines.Count);
            Debug.WriteLine("Number of lines " + lines.Count);
        }
    }
}

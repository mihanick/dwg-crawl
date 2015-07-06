using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Diagnostics;
using Crawl;
using System.Collections.Generic;

namespace TestRectangleTree
{
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
        public void TestSearchTime()
        {
            Rectangle searchedArea = new Rectangle(new crawlPoint3d(0, 0, 0), new crawlPoint3d(350, 150, 0));

            for (int i = 0; i < rectangles.Length; i++)
                rt.Add(rectangles[i]);

            Stopwatch timer = Stopwatch.StartNew();
            List<Rectangle> result = rt.Search(searchedArea);
            timer.Stop();
            Assert.IsTrue(timer.ElapsedMilliseconds < 1);
        }

        [TestMethod]
        public void TestSearchResultSimple()
        {
            Rectangle searchedArea = new Rectangle(new crawlPoint3d(0, 0, 0), new crawlPoint3d(350, 150, 0));

            for (int i = 0; i < rectangles.Length; i++)
                rt.Add(rectangles[i]);

            List<Rectangle> result = rt.Search(searchedArea);

            Assert.AreEqual(2, result.Count);

            CollectionAssert.Contains(result, rectangles[0]);
            CollectionAssert.Contains(result, rectangles[1]);
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
            List<Rectangle> result = rt.Search(searchedArea);
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
            List<Rectangle> result = rt.Search(searchedArea);
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

    [TestClass]
    public class TestRectangle
    {
        private Rectangle[] rectangles;

        public TestRectangle()
        {
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
        public void TestRectangleByTwoPoints()
        {
            crawlPoint3d pntA = new crawlPoint3d(11, 13, 17);
            crawlPoint3d pntC = new crawlPoint3d(19, 23, 29);
            Rectangle rec = new Rectangle(pntA, pntC);
            Assert.AreEqual(rec.pointA, pntA);
            Assert.AreEqual(rec.pointC, pntC);
            Assert.AreEqual(rec.MinPoint, pntA);
            Assert.AreEqual(rec.MaxPoint, pntC);
        }

        [TestMethod]
        public void TestIncludes()
        {
            // Full inclusion
            Assert.IsTrue(rectangles[2].Includes(rectangles[3]));

            // Full non-inclusion
            Assert.IsFalse(rectangles[2].Includes(rectangles[0]));

            // Partial inclusion
            Assert.IsFalse(rectangles[4].Includes(rectangles[6]));
        }

        [TestMethod]
        public void TestIntersects()
        {
            // Partial intersection
            Assert.IsTrue(rectangles[0].Intersects(rectangles[1]));

            // Full intersection
            Assert.IsTrue(rectangles[2].Intersects(rectangles[3]));

            // Side-touching
            Assert.IsTrue(rectangles[4].Intersects(rectangles[5]));

            // No intersection
            Assert.IsFalse(rectangles[0].Intersects(rectangles[5]));

            // No intersection from other side
            Assert.IsFalse(rectangles[5].Intersects(rectangles[0]));
        }
    }

    [TestClass]
    public class TestClusters
    {
        RectangleTree rt;
        Rectangle[] rectangles;

        public TestClusters()
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


            // Clusters to be found (rectangle coordinates and rectangles)
            // at Level0:
            // ((0, 0) (900, 350))
            // rectangles[0..6]

            //     at level1:
            //     ((0, 0) (150, 350))
            //     rectangles[0..1]

            //     ((550, 250) (650, 350))
            //     rectangles[2..3]

            //     ((750, 50) (900, 250))
            //     rectangles[4..6]

            //          at Level2:-
            //          ((575, 275) (625, 325))
            //           rectangles[3]
        }

        [TestMethod]
        public void TestClustersLevel0()
        {
            // Clusters to be found (rectangle coordinates and rectangles)
            // at Level0:
            // ((0, 0) (900, 350))
            // rectangles[0..6]

            foreach (var rect in rectangles)
                rt.Add(rect);

            List<RectangleTree.Cluster> clusters = rt.Clusters(0);

            // Check number of clusters at level 0
            int rectCount = clusters.Count;
            Assert.AreEqual(1, rectCount);

            // Check boundbox of cluster at level 0
            RectangleTree.Cluster cluster0 = clusters[0];
            Rectangle rectLevel0expected = new Rectangle(0, 0, 900, 350);
            Assert.IsTrue(rectLevel0expected.Equals(cluster0.BoundBox));

            // Check resulting contents at level 0
            List<Rectangle> expectedContents = new List<Rectangle>(rectangles);
            foreach (var rect in expectedContents)
                Assert.IsTrue(cluster0.Contains(rect));
        }

        [TestMethod]
        public void TestClustersLevel1()
        {
            //     at level1:
            //     ((0, 0) (150, 350))
            //     rectangles[0..1]

            //     ((550, 250) (650, 350))
            //     rectangles[2..3]

            //     ((750, 50) (900, 250))
            //     rectangles[4..6]


            foreach (var rect in rectangles)
                rt.Add(rect);

            // Getting clusters at level 1
            List<RectangleTree.Cluster> clusters = rt.Clusters(1);

            // Check number of clusters in
            int rectCount = clusters.Count;
            Assert.AreEqual(3, rectCount);

            int grade = 0;
            Rectangle rect1 = new Rectangle(0, 0, 150, 350);
            Rectangle rect2 = new Rectangle(550, 250, 650, 350);
            Rectangle rect3 = new Rectangle(750, 50, 900, 250);

            foreach (RectangleTree.Cluster cluster in clusters)
            {
                // Gain grage++ if current cluster equals one of rectangles

                if (rect1.Equals(cluster.BoundBox))
                {
                    // Check contents match
                    List<Rectangle> expectedContents = new List<Rectangle>();
                    expectedContents.Add(rectangles[0]);
                    expectedContents.Add(rectangles[1]);

                    foreach (var rect in expectedContents)
                        Assert.IsTrue(cluster.Contains(rect));

                    // Increase grade
                    grade++;
                }
                if (rect2.Equals(cluster.BoundBox))
                {
                    // Check contents match
                    List<Rectangle> expectedContents = new List<Rectangle>();
                    expectedContents.Add(rectangles[2]);
                    expectedContents.Add(rectangles[3]);

                    foreach (var rect in expectedContents)
                        Assert.IsTrue(cluster.Contains(rect));

                    // Increase grade
                    grade++;
                }

                if (rect3.Equals(cluster.BoundBox))
                {
                    // Check contents match
                    List<Rectangle> expectedContents = new List<Rectangle>();
                    expectedContents.Add(rectangles[4]);
                    expectedContents.Add(rectangles[5]);
                    expectedContents.Add(rectangles[6]);

                    foreach (var rect in expectedContents)
                        Assert.IsTrue(cluster.Contains(rect));

                    // Increase grade
                    grade++;
                }
            }
            // Check whether we have enough grade to pass
            // If each rectangle in list was met, grade will have 
            Assert.AreEqual(3, grade);
        }

        [TestMethod]
        public void TestClustersLeve2()
        {
            //          at Level2:-
            //          ((575, 275) (625, 325))
            //           rectangles[3]
            foreach (var rect in rectangles)
                rt.Add(rect);

            List<RectangleTree.Cluster> clusters = rt.Clusters(2);

            // Check number of clusters at level 2
            int rectCount = clusters.Count;
            Assert.AreEqual(1, rectCount);

            // Check boundbox of cluster at level 2
            RectangleTree.Cluster cluster = clusters[2];
            Rectangle rectExpected = new Rectangle(575, 275, 625, 325);
            Assert.IsTrue(rectExpected.Equals(cluster.BoundBox));

            // Check resulting contents at level 2
            List<Rectangle> expectedContents = new List<Rectangle>();
            expectedContents.Add(rectangles[3]);

            foreach (var rect in expectedContents)
                Assert.IsTrue(cluster.Contains(rect));
        }
    }
}

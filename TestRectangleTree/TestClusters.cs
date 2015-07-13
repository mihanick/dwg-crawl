namespace Crawl.Test
{
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;

    [TestClass]
    public class TestClusters
    {
        ClusterTree ct;
        Rectangle[] rectangles;

        public TestClusters()
        {
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
        public void TestClustersSmall()
        {
            //     at level1:
            //     ((0, 0) (150, 350))
            //     rectangles[0..1]

            //     ((550, 250) (650, 350))
            //     rectangles[2..3]

            //     ((750, 50) (900, 250))
            //     rectangles[4..6]

            Rectangle[] rectangles1 = new Rectangle[7];
            for (int i = 0; i < rectangles.Length; i++)
                rectangles1[rectangles.Length - i - 1] = rectangles[i];

            ct = new ClusterTree(rectangles1);

            // Getting clusters at level 1
            List<ClusterTree.Cluster> clusters = ct.Clusters;

            // Check number of clusters in
            int rectCount = clusters.Count;
            Assert.AreEqual(4, rectCount);

            int grade = 0;
            Rectangle rect1 = new Rectangle(0, 0, 350, 150);
            Rectangle rect2 = new Rectangle(550, 250, 650, 350);
            Rectangle rect3 = new Rectangle(750, 50, 900, 250);
            Rectangle rect4 = new Rectangle(575, 275, 625, 325);


            foreach (ClusterTree.Cluster cluster in clusters)
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

                if (rect4.Equals(cluster.BoundBox))
                {
                    // Check contents match
                    List<Rectangle> expectedContents = new List<Rectangle>();
                    expectedContents.Add(rectangles[3]);

                    foreach (var rect in expectedContents)
                        Assert.IsTrue(cluster.Contains(rect));

                    // Increase grade
                    grade++;
                }
            }
            // Check whether we have enough grade to pass
            // If each rectangle in list was met, grade will have 
            Assert.AreEqual(4, grade);
        }

        [TestMethod]
        public void TestBigClusterTree()
        {
            SqlDb sqlDB = new SqlDb(@"C:\Data\SingleFile.sdf");
            List<string> jsonOfLines = sqlDB.GetObjectJsonByClassName("AcDbLine");

            List<Rectangle> rects = new List<Rectangle>();

            foreach (string jsonLine in jsonOfLines)
            {
                try
                {
                    crawlAcDbLine cLine = jsonHelper.From<crawlAcDbLine>(jsonLine);
                    if (cLine.Length > 0)
                    {
                        Rectangle rec = new Rectangle(cLine.StartPoint, cLine.EndPoint);
                        rects.Add(rec);
                    }
                }
                catch { }
            }

            Stopwatch timer = Stopwatch.StartNew();
            ClusterTree ct = new ClusterTree(rects.ToArray());
            timer.Stop();

            Assert.IsTrue(timer.ElapsedMilliseconds < 3*rects.Count);

            for (int i = 0; i < ct.Clusters.Count; i++)
                for (int j = 0; j < ct.Clusters.Count; j++)
                {
                    Rectangle rec1 = ct.Clusters[i].BoundBox;
                    Rectangle rec2 = ct.Clusters[j].BoundBox;

                    if (rec1.Equals(rec2))
                        continue;

                    Rectangle notRound1 = new Rectangle(9571.0563, 11257.8221, 12095.1892, 13879.5525);
                    Rectangle notRound2 = new Rectangle(6559.4258, 4018.8264, 16465.4917, 13169.6058);

                    if (Math.Round(rec1.MinPoint.X, 0) == 3704)
                    {
                        //throw new System.ExecutionEngineException();

                        if (Math.Round(rec2.MinPoint.X, 0) == 9571)
                            Debug.WriteLine("Here we should intersect");
                    }

                    if (rec1.Intersects(rec2))
                        Assert.Fail("There's an interesection between clusters");
                }
        }
    }
}
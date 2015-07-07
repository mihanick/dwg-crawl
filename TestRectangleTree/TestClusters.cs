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

            ct = new ClusterTree(rectangles);

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

            List<ClusterTree.Cluster> clusters = ct.Clusters(0);

            // Check number of clusters at level 0
            int rectCount = clusters.Count;
            Assert.AreEqual(1, rectCount);

            // Check boundbox of cluster at level 0
            ClusterTree.Cluster cluster0 = clusters[0];
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

            // Getting clusters at level 1
            List<ClusterTree.Cluster> clusters = ct.Clusters(1);

            // Check number of clusters in
            int rectCount = clusters.Count;
            Assert.AreEqual(3, rectCount);

            int grade = 0;
            Rectangle rect1 = new Rectangle(0, 0, 150, 350);
            Rectangle rect2 = new Rectangle(550, 250, 650, 350);
            Rectangle rect3 = new Rectangle(750, 50, 900, 250);

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

            List<ClusterTree.Cluster> clusters = ct.Clusters(2);

            // Check number of clusters at level 2
            int rectCount = clusters.Count;
            Assert.AreEqual(1, rectCount);

            // Check boundbox of cluster at level 2
            ClusterTree.Cluster cluster = clusters[2];
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
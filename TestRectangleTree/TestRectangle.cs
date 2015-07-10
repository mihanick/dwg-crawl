namespace Crawl.Test
{
    using Microsoft.VisualStudio.TestTools.UnitTesting;

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

            // Touched by side
            Assert.IsFalse(rectangles[4].Includes(rectangles[5]));
        }

        [TestMethod]
        public void TestIntersects()
        {
            // Partial intersection
            Assert.IsTrue(rectangles[0].Intersects(rectangles[1]));

            // Full intersection should not be included
            Assert.IsFalse(rectangles[2].Intersects(rectangles[3]));

            // Intersection of fully included rectangle
            Assert.IsFalse(rectangles[3].Intersects(rectangles[2]));

            // Side-touching
            Assert.IsTrue(rectangles[4].Intersects(rectangles[5]));

            // No intersection
            Assert.IsFalse(rectangles[0].Intersects(rectangles[5]));

            // No intersection backwards
            Assert.IsFalse(rectangles[5].Intersects(rectangles[0]));

            Rectangle notRound1= new Rectangle(9571.0563, 11257.8221, 12095.1892, 13879.5525);
            Rectangle notRound2 = new Rectangle(6559.4258, 4018.8264, 16465.4917, 13169.6058);

            Assert.IsTrue(notRound1.Intersects(notRound2));
        }

        [TestMethod]
        public void TestRectangleIntersection()
        {
            RectangleIntersection ri = new RectangleIntersection(rectangles[0], rectangles[1]);

            Rectangle expectedIntersection = new Rectangle(0,0,350,150);

            Assert.IsTrue(expectedIntersection.Equals(ri));
            Assert.IsTrue(ri.HasIntersection);
        }
    }
}
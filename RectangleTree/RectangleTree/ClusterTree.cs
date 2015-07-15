namespace Crawl
{
    using System.Collections.Generic;
    using System.Diagnostics;

    public class ClusterTree
    {
        // Internal classes
        public class Cluster : HashSet<Rectangle>
        {
            public int Level { get; set; }

            public Rectangle BoundBox { get; set; }

           // public List<Cluster> Children { get; set; }

            public void AddRectangle(Rectangle rec)
            {
                if (rec == null)
                    throw new System.ArgumentNullException("Cannot add null rectangle to cluster");

                if (this.BoundBox == null)
                    this.BoundBox = rec.Clone();

                this.BoundBox = new RectangleIntersection(this.BoundBox, rec);

                base.Add(rec);
            }
        }

        // Properties
        public List<Cluster> Clusters { get; set; }

        private RectangleTree rTree;

        public ClusterTree(Rectangle[] rectangles)
        {
            this.rTree = new RectangleTree(rectangles);
            this.Clusters = new List<Cluster>();

            List<Rectangle> iteratedList = new List<Rectangle>(rectangles);

            while (iteratedList.Count != 0)
            {
                iteratedList = Iterate(iteratedList);
            }

        }

        private List<Rectangle> Iterate(List<Rectangle> input)
        {
            List<Rectangle> result = new List<Rectangle>(input);
            HashSet<Rectangle> toDelete = new HashSet<Rectangle>();

            toDelete = RecursiveIntersections(input[0], toDelete);

            Cluster cluster = new Cluster();
            foreach (Rectangle rect in toDelete)
                cluster.AddRectangle(rect);
            this.AddCluster(cluster);

            foreach (var rec in toDelete)
                result.Remove(rec);
            return result;
        }

        private void AddCluster(Cluster clusterNew)
        {
            foreach (Cluster cluster in this.Clusters)
                if (cluster.BoundBox.Intersects(clusterNew.BoundBox))
                {
                    foreach (Rectangle rec in clusterNew)
                    {
                        cluster.AddRectangle(rec);
                    }
                    this.Clusters.Remove(cluster);
                    this.AddCluster(cluster);
                    return;
                }

            this.Clusters.Add(clusterNew);
        }

        private HashSet<Rectangle> RecursiveIntersections(Rectangle rec, HashSet<Rectangle> allIntersections)
        {
            HashSet<Rectangle> result = new HashSet<Rectangle>();
            foreach (var rect in allIntersections)
                result.Add(rect);

            result.Add(rec);

            foreach (Rectangle rect in rTree.Intersections(rec))
            {
                if (!result.Contains(rect))
                    result = RecursiveIntersections(rect, result);
            }

            return result;
        }

        public List<Cluster> ClustersAtLevel(int level)
        {
            List<Cluster> result = new List<Cluster>();
            foreach (var cluster in this.Clusters)
                if (cluster.Level == level)
                    result.Add(cluster);

            return this.Clusters;
        }
    }
}

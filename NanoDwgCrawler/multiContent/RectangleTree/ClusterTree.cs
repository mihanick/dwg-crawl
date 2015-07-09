namespace Crawl
{
    using System.Collections.Generic;

    public class ClusterTree
    {
        private RectangleTree rTree;

        public int Depth { get; set; }

        List<Cluster> clusters;

        #region Internal Classes

        public class Cluster : HashSet<Rectangle>
        {
            #region properties
            public int Level { get; set; }

            private Rectangle boundBox;
            public Rectangle BoundBox
            {
                get
                {
                    return this.boundBox;
                }
            }
            #endregion

            public Cluster()
            {
                this.Level = -1;
            }

            #region Methods

            public void Add(Rectangle rec)
            {
                if (rec == null)
                    throw new System.NullReferenceException("Cannot add null rectangle");

                base.Add(rec);

                // First-time cluster initialisation from first rectangle
                if (this.boundBox == null)
                    this.boundBox = rec.Clone();

                RectangleIntersection intersection = new RectangleIntersection(this.boundBox, rec);
                this.boundBox = intersection;
            }
            #endregion
        }
        #endregion

        public ClusterTree(Rectangle[] rects)
        {
            this.clusters = new List<Cluster>();
            this.rTree = new RectangleTree();
            List<Rectangle> rectangles = new List<Rectangle>(rects);

            foreach (var rec in rects)
                this.rTree.Add(rec);

            this.FindClusters(rectangles);
        }

        private void FindClusters(List<Rectangle> rectangles)
        {
            List<Rectangle> iteratedList = new List<Rectangle>(rectangles);

            while (iteratedList.Count != 0)
            {
                iteratedList = iterate(iteratedList);
            }
        }

        private List<Rectangle> iterate(List<Rectangle> rectangles)
        {
            List<Rectangle> input = new List<Rectangle>(rectangles);

            List<Rectangle> toRemove = new List<Rectangle>();

            foreach (Rectangle rec in input)
            {
                List<Rectangle> intersections = rTree.Intersections(rec);
                toRemove.AddRange(intersections);

                // Поскольку rTree.Intersections и так должно возвращать сам прямоугольник rec, то его не нужно добавлять повторно
                // toRemove.Add(rec);

                foreach (Rectangle rectangleIntersected in toRemove)
                {
                    input.Remove(rectangleIntersected);

                    AddToClusterOrCreate(rectangleIntersected);
                }

                List<Rectangle> inclusions = rTree.Inclusions(rec);

                foreach (Rectangle rectangleIncluded in inclusions)
                {
                    input.Remove(rectangleIncluded);
                    AddIncluded(rectangleIncluded);
                }

                return input;
            }

            return new List<Rectangle>();
        }

        private void AddIncluded(Rectangle rec)
        {
            foreach (Cluster cluster in this.clusters)
                foreach (Rectangle rectBig in cluster)
                    if (rectBig.Includes(rec))
                    {
                        Cluster clusterNew = new Cluster();
                        clusterNew.Level = cluster.Level + 1;
                        clusterNew.Add(rec);

                        // We want included rectangles to be included in parent cluster as well
                        cluster.Add(rec);

                        clusters.Add(clusterNew);

                        return;
                    }
        }

        private void AddToClusterOrCreate(Rectangle rec)
        {
            Cluster clusterToAdd = new Cluster();
            clusterToAdd.Level = 0;

            if (this.clusters.Count == 0)
            {
                this.clusters.Add(clusterToAdd);
            }
            else
            {
                bool needToAddNewCluster = true;

                foreach (Cluster cluster in this.clusters)
                    if (cluster.BoundBox.Intersects(rec))
                    {
                        clusterToAdd = cluster;
                        // Cluster found, no need to add new
                        needToAddNewCluster = false;
                        break;
                    }

                // If cluster wasn't found, we should add new cluster to list of clusters
                if (needToAddNewCluster)
                    this.clusters.Add(clusterToAdd);
            }
            clusterToAdd.Add(rec);
        }

        #region Methods

        public List<Cluster> Clusters(int level = 0)
        {
            List<Cluster> result = new List<Cluster>();
            foreach (Cluster cluster in this.clusters)
                if (cluster.Level == level)
                    result.Add(cluster);

            return result;
        }
        #endregion
    }
}

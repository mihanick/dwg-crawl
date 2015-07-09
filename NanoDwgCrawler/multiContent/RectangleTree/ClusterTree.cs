namespace Crawl
{
    using System.Collections.Generic;
    using System.Diagnostics;

    public class ClusterTree
    {
        private RectangleTree rTree;

        public int Depth { get; set; }

        public List<Cluster> clusters;

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
                Stopwatch timer = Stopwatch.StartNew();
                iteratedList = this.iterate(iteratedList);
                timer.Stop();
                Debug.WriteLine(iteratedList.Count.ToString()+" "+timer.ElapsedMilliseconds.ToString());
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

                // TODO: WTF?? Should include itself by search
                if (!toRemove.Contains(rec))
                    toRemove.Add(rec);

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

            JoinClusters();
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
                    if (cluster.BoundBox.Intersects(rec) || cluster.BoundBox.Includes(rec))
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

            JoinClusters();
        }

        private void JoinClusters()
        {
            bool cntinue = false;
            while (cntinue)
                cntinue = this.iterateClusters(this.clusters);
        }

        private bool iterateClusters(List<Cluster> inputlist)
        {
            foreach (Cluster cluster in this.clusters)
                foreach (Cluster otherCluster in this.clusters)
                    if (cluster != otherCluster)
                    {
                        if (cluster.BoundBox.Intersects(otherCluster.BoundBox))
                        {
                            foreach (Rectangle rec in otherCluster)
                                cluster.Add(rec);
                            this.clusters.Remove(otherCluster);
                            return true;
                        }
                        if (cluster.BoundBox.Includes(otherCluster.BoundBox))
                        {
                            otherCluster.Level = cluster.Level + 1;
                        }
                    }

            return false;
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

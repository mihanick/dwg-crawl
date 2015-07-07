using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Crawl;

namespace Crawl
{
    public class RectangleTree
    {
        public Node Root { get; set; }

        #region Internal Classes
        public class Node
        {
            private Rectangle contents;
            public Rectangle Contents
            {
                get
                {
                    return this.contents;
                }
            }

            private Rectangle boundBox;
            public Rectangle BoundBox
            {
                get
                {
                    return this.boundBox;
                }
            }

            public Node()
            {
                this.boundBox = new Rectangle();
                this.children = new List<Node>();
                this.boundBox = new Rectangle();
            }

            public Node(Rectangle rec)
            {
                this.contents = rec;
                this.boundBox = rec.Clone();
                this.children = new List<Node>();
            }

            public Node(Node nodeLeft, Node nodeRight)
            {
                this.children = new List<Node>();
                this.children.Add(nodeLeft);
                this.children.Add(nodeRight);

                this.contents = null;
                this.boundBox = new RectangleIntersection(nodeLeft.boundBox, nodeRight.boundBox);

            }

            private List<Node> children;
            public List<Node> Children
            {
                get
                {
                    return this.children;
                }
            }

            public void AddChild(Node node)
            {
                foreach (Node child in this.children)
                    if (child.Contains(node.boundBox))
                    {
                        child.AddChild(node);
                        return;
                    }

                this.children.Add(node);
                RectangleIntersection intersection = new RectangleIntersection(this.boundBox, node.BoundBox);
                this.boundBox = intersection;
            }

            public bool Contains(Rectangle rec)
            {
                if (this.boundBox.Includes(rec))
                    return true;

                return false;
            }

            public List<Rectangle> Included(Rectangle rec)
            {
                List<Rectangle> result = new List<Rectangle>();

                if (this.contents == null)
                {
                    foreach (Node child in this.children)
                        result.AddRange(child.Included(rec));
                }
                else
                    if (rec.Includes(this.boundBox))
                        result.Add(this.contents);

                return result;
            }

            public List<Rectangle> Intersected(Rectangle rec)
            {
                List<Rectangle> result = new List<Rectangle>();

                if (this.contents == null)
                {
                    foreach (Node child in this.children)
                        result.AddRange(child.Included(rec));
                }
                else
                    if (rec.Intersects(this.boundBox))
                        result.Add(this.contents);

                return result;
            }
        }

        #endregion

        public RectangleTree()
        {
        }

        #region Methods
        public void Add(Rectangle rec)
        {
            if (this.Root == null)
                this.Root = new Node(rec);
            else
                AddContents(rec);

        }

        private void AddContents(Rectangle rec)
        {
            if (this.Root == null)
                throw new Exception("Нет корневого элемента");

            if (rec == null)
                throw new Exception("Нельзя добавлять пустой прямоугольник");

            // Если добавляемый прямоугольник не содержится в корневом элементе
            if (!this.Root.Contains(rec))
            {
                Node recNode = new Node(rec);
                // То создаем новый корневой элемент, который будет содержать два элемента
                this.Root = new Node(this.Root, recNode);
            }
            else
            {
                this.Root.AddChild(new Node(rec));
            }
        }

        public List<Rectangle> Search(Rectangle rec)
        {
            return Root.Included(rec);
        }

        public List<Rectangle> Intersections(Rectangle rec)
        {
            return this.Root.Intersected(rec);
        }

        #endregion
    }

    public class ClusterTree
    {
        private RectangleTree rTree;

        public int Depth { get; set; }

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
                this.boundBox = new Rectangle();
            }

            #region Methods

            public void Add(Rectangle rec)
            {
                base.Add(rec);
                RectangleIntersection intersection = new RectangleIntersection(this.boundBox, rec);
                this.boundBox = intersection;
            }
            #endregion
        }
        #endregion

        public ClusterTree(Rectangle[] rects)
        {
            List<Rectangle> rectangles = new List<Rectangle>(rects);
            foreach (var rec in rects)
                this.rTree.Add(rec);

            this.FindClusters(rectangles);
        }

        private void FindClusters(List<Rectangle> rectangles)
        {
            List<Cluster> clusters = new List<Cluster>();

            List<Rectangle> iteratedList = new List<Rectangle>(rectangles);

            while (iteratedList.Count != 0)
            {
                iteratedList = iterate(iteratedList);
            }
        }

        private List<Rectangle> iterate(List<Rectangle> rectangles)
        {
            List<Rectangle> input = new List<Rectangle>(rectangles);
            List<Cluster> clusters = new List<Cluster>();
            List<Rectangle> toRemove = new List<Rectangle>();

            foreach (Rectangle rec in input)
            {
                List<Rectangle> intersections = rTree.Intersections(rec);
                toRemove.AddRange(intersections);

                Cluster cluster = new Cluster();
                foreach (Rectangle rectangleIntersected in intersections)
                    cluster.Add(rectangleIntersected);

                clusters.Add(cluster);
            }

            foreach (Rectangle rect in toRemove)
                input.Remove(rect);

            return input;
        }

        #region Methods

        public List<Cluster> Clusters(int level = 0)
        {
            throw new NotImplementedException();
        }
        #endregion
    }
}

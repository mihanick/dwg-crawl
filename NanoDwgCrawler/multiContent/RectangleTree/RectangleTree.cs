namespace Crawl
{
    using System;
    using System.Collections.Generic;

    public class RectangleTree
    {
        public Node Root { get; set; }

        #region Internal Classes
        public class Node
        {
            #region properties
            public bool IsALeaf;

            private Rectangle contents;
            public Rectangle Contents
            {
                get
                {
                    return this.contents;
                }
            }

            private List<Node> children;
            public List<Node> Children
            {
                get
                {
                    return this.children;
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
            #endregion

            #region Constructors
            public Node(Node childLeaf)
            {
                this.IsALeaf = false;
                this.children = new List<Node>();
                this.children.Add(childLeaf);
                this.boundBox = childLeaf.boundBox.Clone();
            }

            public Node(Rectangle rec)
            {
                this.IsALeaf = true;
                this.contents = rec;
                this.boundBox = rec.Clone();
                this.children = new List<Node>();
            }

            public Node(Node nodeLeft, Node nodeRight)
            {
                this.IsALeaf = false;
                this.children = new List<Node>();
                this.children.Add(nodeLeft);
                this.children.Add(nodeRight);

                this.contents = null;
                this.boundBox = new RectangleIntersection(nodeLeft.boundBox, nodeRight.boundBox);
            }
            #endregion

            #region Methods
            public void AddChild(Node node)
            {
                if (node == null)
                    throw new ArgumentNullException();

                if (this.IsALeaf)
                    throw new System.Exception(" Cannot add child to a leaf");

                foreach (Node child in this.children)
                    if (!child.IsALeaf)
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
                if (this.boundBox.Intersects(rec) || this.boundBox.Includes(rec))
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
                        result.AddRange(child.Intersected(rec));
                }
                else
                    if (rec.Intersects(this.boundBox))
                        result.Add(this.contents);

                return result;
            }

            public override string ToString()
            {
                return this.ConvertToString();
            }

            public string ConvertToString(string indent = "")
            {
                string result = indent + this.boundBox.ToString() + Environment.NewLine;
                foreach (Node child in this.children)
                    result += child.ConvertToString(indent + "\t") + Environment.NewLine;
                if (this.IsALeaf)
                    result += indent+"\t" + "rec:" + this.contents.ToString();
                return result;
            }

            public List<Rectangle> GetContents()
            {
                List<Rectangle> result = new List<Rectangle>();
                foreach (Node child in this.children)
                    result.AddRange(child.GetContents());

                if (this.contents != null)
                    result.Add(this.contents);

                return result;
            }
            #endregion
        }

        #endregion

        public RectangleTree()
        {
            //this.Root = new Node();
        }

        #region Methods
        public void Add(Rectangle rec)
        {
            AddContents(rec);
        }

        private void AddContents(Rectangle rec)
        {
            if (rec == null)
                throw new NullReferenceException("Нельзя добавлять пустой прямоугольник");

            // If it's the first rectangle to add
            if (this.Root == null)
            {
                Node firstLeaf= new Node(rec);
                this.Root = new Node(firstLeaf);
                return;
            }

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

        public List<Rectangle> Inclusions(Rectangle rec)
        {
            return Root.Included(rec);
        }

        public List<Rectangle> Intersections(Rectangle rec)
        {
            return this.Root.Intersected(rec);
        }

        public List<Rectangle> GetContents()
        {
            return this.Root.GetContents();
        }
        #endregion
    }
}

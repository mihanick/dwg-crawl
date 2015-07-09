namespace Crawl
{
    using System;

    /// <summary>
    /// Represents rectangle ABCD counter-clockwise
    /// </summary>
    public class Rectangle
    {
        /*
        public crawlAcDbLine line1;
        public crawlAcDbLine line2;
        public crawlAcDbLine line3;
        public crawlAcDbLine line4;

        public Rectangle Rectangle1;
        public Rectangle Rectangle2;

        public double IntersectionAreaPercent1
        {
            get
            {
                return this.Area / this.Rectangle1.Area;
            }
        }

        public double IntersectionAreaPercent2
        {
            get
            {
                return this.Area / this.Rectangle2.Area;
            }
        }
       */

        public crawlPoint3d pointA;
        public crawlPoint3d pointC;

        public crawlPoint3d MinPoint
        {
            get
            {
                if (pointA.X < pointC.X)
                    return pointA;
                if (pointA.X > pointC.X)
                    return pointC;
                if (pointA.Y < pointC.Y)
                    return pointA;
                if (pointA.Y > pointC.Y)
                    return pointC;
                if (pointA.Z < pointC.Z)
                    return pointA;
                if (pointA.Z > pointC.Z)
                    return pointC;
                return pointA;
            }
        }

        public crawlPoint3d MaxPoint
        {
            get
            {
                if (pointA.X > pointC.X)
                    return pointA;
                if (pointA.X < pointC.X)
                    return pointC;
                if (pointA.Y > pointC.Y)
                    return pointA;
                if (pointA.Y < pointC.Y)
                    return pointC;
                if (pointA.Z > pointC.Z)
                    return pointA;
                if (pointA.Z < pointC.Z)
                    return pointC;
                return pointA;
            }
        }

        public double Length
        {
            get
            {
                return this.MaxPoint.X - this.MinPoint.X;
            }
        }

        public double Height
        {
            get
            {
                return this.MaxPoint.Y - this.MinPoint.Y;
            }
        }

        public double Area
        {
            get
            {
                return this.Length * this.Height;
            }
        }

        public double Perimeter
        {
            get
            {
                return 2 * (this.Length + this.Height);
            }
        }

        #region Constructors
        
        public Rectangle()
        {
            // Required by child classes
        }
        

        public Rectangle(double pointAx, double pointAy, double pointCx, double pointCy)
        {
            this.pointA = new crawlPoint3d(pointAx, pointAy, 0);
            this.pointC = new crawlPoint3d(pointCx, pointCy, 0);
        }

        public Rectangle(crawlPoint3d bottomLeftCorner, crawlPoint3d topRightCorner)
        {
            this.pointA = bottomLeftCorner;
            this.pointC = topRightCorner;
        }

        #endregion

        #region Methods
        /// <summary>
        /// Checks whether Rectangle rec is fully included in this rectangle
        /// </summary>
        /// <param name="rec">Rectangle to check inclusion</param>
        /// <returns>true if rec is fully included </returns>
        public bool Includes(Rectangle rec)
        {
            if (this.MinPoint.X >= rec.MinPoint.X)
                return false;
            if (this.MinPoint.Y >= rec.MinPoint.Y)
                return false;

            if (this.MaxPoint.X <= rec.MaxPoint.X)
                return false;
            if (this.MaxPoint.Y <= rec.MaxPoint.Y)
                return false;

            return true;
        }

        /// <summary>
        /// Checks whether Rectangle rec intersects this rectangle, but not included
        /// </summary>
        /// <param name="rec">Rectangle to check intersection</param>
        /// <returns>true if rec at least touches this rectangle</returns>
        public bool Intersects(Rectangle rec)
        {
            if (this.MinPoint.X > rec.MaxPoint.X)
                return false;
            if (this.MinPoint.Y > rec.MaxPoint.Y)
                return false;

            if (this.MaxPoint.X < rec.MinPoint.X)
                return false;
            if (this.MaxPoint.Y < rec.MinPoint.Y)
                return false;

            // Special check that it is not full inclusion
            if (this.Includes(rec))
                return false;

            return true;
        }

        /// <summary>
        /// Checks whether point is inside rectangle
        /// </summary>
        /// <param name="pnt">Point to check</param>
        /// <returns>True if point at least on a border</returns>
        public bool Includes(crawlPoint3d pnt)
        {
            if (this.MinPoint.X > pnt.X)
                return false;
            if (this.MinPoint.Y > pnt.Y)
                return false;
            if (this.MinPoint.Z > pnt.Z)
                return false;

            if (this.MaxPoint.X < pnt.X)
                return false;
            if (this.MaxPoint.Y < pnt.Y)
                return false;
            if (this.MaxPoint.Z < pnt.Z)
                return false;

            return true;
        }

        public Rectangle Clone()
        {
            return new Rectangle(this.pointA, this.pointC);
        }

        public bool Equals(Rectangle otherRectangle)
        {
            // Recomendations for implementing Equals
            // https://msdn.microsoft.com/en-US/library/ms173147(v=vs.80).aspx

            if (otherRectangle == null)
                return false;
            if (this == otherRectangle)
                return true;

            if (this.pointA.Equals(otherRectangle.pointA) && this.pointC.Equals(otherRectangle.pointC))
                return true;

            return false;
        }

        public override string ToString()
        {
            return string.Format("({0}, {1}), ({2}, {3})", this.MinPoint.X.ToString(), this.MinPoint.Y.ToString(), this.MaxPoint.X.ToString(), this.MaxPoint.Y.ToString());
        }
        #endregion
    }
    
    public class RectangleIntersection : Rectangle
    {
        public Rectangle Rectangle1;
        public Rectangle Rectangle2;

        public bool Intersects;
        public double IntersectionArea;


        public double IntersectionAreaPercent1
        {
            get
            {
                return this.IntersectionArea / this.Rectangle1.Area;
            }
        }

        public double IntersectionAreaPercent2
        {
            get
            {
                return this.IntersectionArea / this.Rectangle2.Area;
            }
        }


        #region Constructors

        public RectangleIntersection(Rectangle rectangle1, Rectangle rectangle2)
        {
            this.Rectangle1 = rectangle1;
            this.Rectangle2 = rectangle2;

            if (this.Rectangle1.MaxPoint.X < this.Rectangle2.MinPoint.X ||
                this.Rectangle2.MaxPoint.X < this.Rectangle1.MinPoint.X ||
                this.Rectangle1.MaxPoint.Y < this.Rectangle2.MinPoint.Y ||
                this.Rectangle2.MaxPoint.Y < this.Rectangle1.MinPoint.Y)
            {
                this.Intersects = false;

                double minX = Math.Min(this.Rectangle1.MinPoint.X, this.Rectangle2.MinPoint.X);
                double maxX = Math.Max(this.Rectangle1.MaxPoint.X, this.Rectangle2.MaxPoint.X);
                double minY = Math.Min(this.Rectangle1.MinPoint.Y, this.Rectangle2.MinPoint.Y);
                double maxY = Math.Max(this.Rectangle1.MaxPoint.Y, this.Rectangle2.MaxPoint.Y);

                this.pointA = new crawlPoint3d(minX, minY, 0);
                this.pointC = new crawlPoint3d(maxX, maxY, 0);

                this.IntersectionArea = this.Area;
            }
            else
            {
                double minX = Math.Min(this.Rectangle1.MinPoint.X, this.Rectangle2.MinPoint.X);
                double maxX = Math.Max(this.Rectangle1.MaxPoint.X, this.Rectangle2.MaxPoint.X);
                double minY = Math.Min(this.Rectangle1.MinPoint.Y, this.Rectangle2.MinPoint.Y);
                double maxY = Math.Max(this.Rectangle1.MaxPoint.Y, this.Rectangle2.MaxPoint.Y);

                this.pointA = new crawlPoint3d(minX, minY, 0);
                this.pointC = new crawlPoint3d(maxX, maxY, 0);

                this.Intersects = true;

                double minXi = Math.Max(this.Rectangle1.MinPoint.X, this.Rectangle2.MinPoint.X);
                double maxXi = Math.Min(this.Rectangle1.MaxPoint.X, this.Rectangle2.MaxPoint.X);
                double minYi = Math.Max(this.Rectangle1.MinPoint.Y, this.Rectangle2.MinPoint.Y);
                double maxYi = Math.Min(this.Rectangle1.MaxPoint.Y, this.Rectangle2.MaxPoint.Y);

                crawlPoint3d intersectionA = new crawlPoint3d(minXi, minYi, 0);
                crawlPoint3d intersectionC = new crawlPoint3d(maxXi, maxYi, 0);

            }
        }
        #endregion

        #region Methods

        #endregion
    }
   
}
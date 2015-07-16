namespace Crawl
{
    using System;
    using System.Diagnostics;
    using System.Globalization;
    using System.Runtime.Serialization;

    [DataContract]
    public class crawlPoint3d
    {
        [DataMember]
        string ClassName = "Point3D";

        [DataMember]
        public double X;
        [DataMember]
        public double Y;
        [DataMember]
        public double Z;

        public crawlPoint3d()
        {
            this.X = 0;
            this.Y = 0;
            this.Z = 0;
        }

        public crawlPoint3d(double X, double Y, double Z)
        {
            this.X = X;
            this.Y = Y;
            this.Z = Z;
        }

        public override string ToString()
        {
            return string.Format("({0}, {1}, {2})", Math.Round(this.X, 2), Math.Round(this.Y, 2), Math.Round(this.Z, 2));
        }

        public bool Equals(crawlPoint3d otherPoint3d)
        {
            return this.ToString().Equals(otherPoint3d.ToString());
        }
    }

    /// <summary>
    /// Represents rectangle ABCD counter-clockwise
    /// </summary>
    public class Rectangle
    {

        public crawlPoint3d pointA;
        public crawlPoint3d pointC;

        public crawlPoint3d MinPoint
        {
            get
            {
                double x = Math.Min(pointA.X, pointC.X);
                double y = Math.Min(pointA.Y, pointC.Y);
                double z = Math.Min(pointA.Z, pointC.Z);

                /*
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
                 */
                return new crawlPoint3d(x,y,z);
            }
        }

        public crawlPoint3d MaxPoint
        {
            get
            {
                double x = Math.Max(pointA.X, pointC.X);
                double y = Math.Max(pointA.Y, pointC.Y);
                double z = Math.Max(pointA.Z, pointC.Z);

                return new crawlPoint3d(x, y, z);
                /*
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
                 */
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

        /// <summary>
        /// Initializes a Rectangle from the string of coordinates
        /// </summary>
        /// <param name="stringCoords">string like 'x1;y1;z1;x2;y2;z2'</param>
        public Rectangle(string stringCoords)
        {
            try
            {
                string[] xyz = stringCoords.Split(';');
                if (xyz.Length != 6)
                    Debug.WriteLine("Wrong input in rectangle constructor from string coordinates");

                double x1;
                TryParse(xyz[0],out x1);

                double y1;
                TryParse(xyz[1], out y1);

                double x2;
                TryParse(xyz[3], out x2);

                double y2;
                TryParse(xyz[4], out y2);

                this.pointA = new crawlPoint3d(x1, y1, 0);
                this.pointC = new crawlPoint3d(x2, y2, 0);
            }
            catch
            {
                // Initialization failed
            }
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
            if (this.Includes(rec) || rec.Includes(this))
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

        /// <summary>
        /// Функция парсит строку в double с учетом текущих настроек десятчиного разделителя
        /// </summary>
        /// <param name="stringValue">Исходная строка</param>
        /// <param name="doubleValue">Возвращаемое значение, если не удалось преобразовать - будет 0.0</param>
        /// <returns>true, если исходная строка может быть преобразована к double</returns>
        private static bool TryParse(string stringValue, out double doubleValue)
        {
            string trimmed = stringValue.Trim();

            double l2 = 0.0;
            if (double.TryParse(trimmed, System.Globalization.NumberStyles.Float, CultureInfo.InvariantCulture, out l2)
                 || double.TryParse(trimmed, NumberStyles.Float, CultureInfo.CurrentCulture, out l2)
                    || double.TryParse(trimmed.Replace(',', '.'), NumberStyles.Float, CultureInfo.InvariantCulture, out l2))
            {
                doubleValue = l2;
                return true;
            }
            else
            {
                doubleValue = l2;
                return false;
            }
        }

        #endregion
    }
    
    public class RectangleIntersection : Rectangle
    {
        public Rectangle Rectangle1;
        public Rectangle Rectangle2;

        public bool HasIntersection;
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

            double minX = Math.Min(this.Rectangle1.MinPoint.X, this.Rectangle2.MinPoint.X);
            double maxX = Math.Max(this.Rectangle1.MaxPoint.X, this.Rectangle2.MaxPoint.X);
            double minY = Math.Min(this.Rectangle1.MinPoint.Y, this.Rectangle2.MinPoint.Y);
            double maxY = Math.Max(this.Rectangle1.MaxPoint.Y, this.Rectangle2.MaxPoint.Y);


            if (this.Rectangle1.MaxPoint.X < this.Rectangle2.MinPoint.X ||
                this.Rectangle2.MaxPoint.X < this.Rectangle1.MinPoint.X ||
                this.Rectangle1.MaxPoint.Y < this.Rectangle2.MinPoint.Y ||
                this.Rectangle2.MaxPoint.Y < this.Rectangle1.MinPoint.Y)
            {
                this.HasIntersection = false;

                this.pointA = new crawlPoint3d(minX, minY, 0);
                this.pointC = new crawlPoint3d(maxX, maxY, 0);

                this.IntersectionArea = this.Area;
            }
            else
            {
                this.pointA = new crawlPoint3d(minX, minY, 0);
                this.pointC = new crawlPoint3d(maxX, maxY, 0);

                this.HasIntersection = true;

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
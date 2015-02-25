using System.Runtime.Serialization;
using Teigha.Geometry;
using Teigha.DatabaseServices;
using System.Collections.Generic;
using System;
using Crawl;

[DataContract]
public class crawlAcDbLine
{
    [DataMember]
    string ClassName = "AcDbLine";
    [DataMember]
    string Color;
    
    [DataMember]
    public crawlPoint3d EndPoint { get; set; }
    [DataMember]
    public crawlPoint3d StartPoint { get; set; }
    [DataMember]
    public double Length;
    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbLine(Line line)
    {
        this.EndPoint = new crawlPoint3d(line.EndPoint);
        this.StartPoint = new crawlPoint3d(line.StartPoint);
        this.Layer = line.Layer;
        this.Linetype = line.Linetype;
        this.LineWeight = line.LineWeight.ToString();
        this.Color = line.Color.ToString();

        this.Length = line.Length;
    }
}

[DataContract]
public class crawlAcDbPolyline
{
    [DataMember]
    string ClassName = "AcDbPolyline";
    [DataMember]
    string Color;

    [DataMember]
    double Length;
    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;
    [DataMember]
    double Area;

    [DataMember]
    List<crawlPoint3d> Vertixes;

    public crawlAcDbPolyline(Polyline polyline)
    {
        this.Length = polyline.Length;
        this.Area = polyline.Area;

        this.Layer = polyline.Layer;
        this.Linetype = polyline.Linetype;
        this.LineWeight = polyline.LineWeight.ToString();
        this.Color = polyline.Color.ToString();

        Vertixes = new List<crawlPoint3d>();

        // Use a for loop to get each vertex, one by one
        int vn = polyline.NumberOfVertices;
        for (int i = 0; i < vn; i++)
        {
            Vertixes.Add(new crawlPoint3d(polyline.GetPoint3dAt(i)));
        }
    }

    public crawlAcDbPolyline(Polyline2d polyline)
    {
        Length = polyline.Length;
        this.Layer = polyline.Layer;
        this.Linetype = polyline.Linetype;
        this.LineWeight = polyline.LineWeight.ToString();
        this.Color = polyline.Color.ToString();

        Vertixes = new List<crawlPoint3d>();

        // Use foreach to get each contained vertex
        foreach (ObjectId vId in polyline)
        {
            Vertex2d v2d =
              (Vertex2d)
                vId.GetObject(
                OpenMode.ForRead
              );
            Vertixes.Add(new crawlPoint3d(v2d.Position));
        }
    }

    public crawlAcDbPolyline(Polyline3d polyline)
    {
        Length = polyline.Length;
        this.Layer = polyline.Layer;
        this.Linetype = polyline.Linetype;
        this.LineWeight = polyline.LineWeight.ToString();
        this.Color = polyline.Color.ToString();

        Vertixes = new List<crawlPoint3d>();

        // Use foreach to get each contained vertex
        foreach (ObjectId vId in polyline)
        {
            PolylineVertex3d v3d =
              (PolylineVertex3d)
                vId.GetObject(OpenMode.ForRead);
            Vertixes.Add(new crawlPoint3d(v3d.Position));
        }
    }
}

[DataContract]
public class crawlAcDbText
{
    [DataMember]
    string ClassName = "AcDbText";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d Position { get; set; }
    [DataMember]
    public string TextString;
    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbText(DBText text)
    {
        this.Position = new crawlPoint3d(text.Position);

        this.Layer = text.Layer;
        this.Linetype = text.Linetype;
        this.LineWeight = text.LineWeight.ToString();
        this.Color = text.Color.ToString();

        this.TextString = text.TextString;
    }
}

[DataContract]
public class crawlAcDbMText
{
    [DataMember]
    string ClassName = "AcDbMText";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d Position { get; set; }
    [DataMember]
    public string TextString;
    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbMText(MText text)
    {
        this.Position = new crawlPoint3d(text.Location);
        this.Layer = text.Layer;
        this.Linetype = text.Linetype;
        this.LineWeight = text.LineWeight.ToString();
        this.Color = text.Color.ToString();

        this.TextString = text.Contents;
    }
}

[DataContract]
public class crawlAcDbAttributeDefinition
{
    [DataMember]
    string ClassName = "AcDbAttributeDefinition";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d Position { get; set; }
    [DataMember]
    public string TextString;

    [DataMember]
    public string Prompt;
    [DataMember]
    public string Tag;

    [DataMember]
    public crawlAcDbMText MTextAttributeDefinition;

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbAttributeDefinition(AttributeDefinition att)
    {
        this.Position = new crawlPoint3d(att.Position);
        this.Layer = att.Layer;
        this.Linetype = att.Linetype;
        this.LineWeight = att.LineWeight.ToString();
        this.Color = att.Color.ToString();

        this.Prompt = att.Prompt;
        this.Tag = att.Tag;

        this.MTextAttributeDefinition = new crawlAcDbMText(att.MTextAttributeDefinition);
    }
}

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
    public crawlPoint3d(Point3d pt)
    {
        this.X = pt.X;
        this.Y = pt.Y;
        this.Z = pt.Z;
    }

}

[DataContract]
public class crawlAcDbArc
{
    [DataMember]
    string ClassName = "AcDbArc";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d Center { get; set; }
    [DataMember]
    public crawlPoint3d StartPoint { get; set; }
    [DataMember]
    public crawlPoint3d EndPoint { get; set; }

    [DataMember]
    public double Length;
    [DataMember]
    public double Thickness;
    [DataMember]
    public double Radius;

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbArc(Arc arc)
    {
        this.EndPoint = new crawlPoint3d(arc.EndPoint);
        this.StartPoint = new crawlPoint3d(arc.StartPoint);
        this.Center = new crawlPoint3d(arc.Center);

        this.Layer = arc.Layer;
        this.Linetype = arc.Linetype;
        this.LineWeight = arc.LineWeight.ToString();
        this.Color = arc.Color.ToString();

        this.Length = arc.Length;
        this.Thickness = arc.Thickness;

        this.Radius = arc.Radius;
    }
}

[DataContract]
public class crawlAcDbCircle
{
    [DataMember]
    string ClassName = "AcDbCircle";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d Center { get; set; }
    [DataMember]
    public crawlPoint3d StartPoint { get; set; }
    [DataMember]
    public crawlPoint3d EndPoint { get; set; }

    [DataMember]
    public double Length;
    [DataMember]
    public double Thickness;
    [DataMember]
    public double Radius;

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbCircle(Circle circle)
    {
        this.EndPoint = new crawlPoint3d(circle.EndPoint);
        this.StartPoint = new crawlPoint3d(circle.StartPoint);
        this.Center = new crawlPoint3d(circle.Center);

        this.Layer = circle.Layer;
        this.Linetype = circle.Linetype;
        this.LineWeight = circle.LineWeight.ToString();
        this.Color = circle.Color.ToString();

        this.Radius = circle.Radius;
        this.Thickness = circle.Thickness;
    }
}

[DataContract]
public class crawlAcDbEllipse
{
    [DataMember]
    string ClassName = "AcDbEllipse";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d Center { get; set; }
    [DataMember]
    public crawlPoint3d StartPoint { get; set; }
    [DataMember]
    public crawlPoint3d EndPoint { get; set; }

    [DataMember]
    public double Length;

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbEllipse(Ellipse ellipse)
    {
        this.EndPoint = new crawlPoint3d(ellipse.EndPoint);
        this.StartPoint = new crawlPoint3d(ellipse.StartPoint);
        this.Center = new crawlPoint3d(ellipse.Center);

        this.Layer = ellipse.Layer;
        this.Linetype = ellipse.Linetype;
        this.LineWeight = ellipse.LineWeight.ToString();
        this.Color = ellipse.Color.ToString();

    }
}

[DataContract]
public class crawlAcDbRotatedDimension
{
    [DataMember]
    string ClassName = "AcDbRotatedDimension";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d XLine1Point { get; set; }
    [DataMember]
    public crawlPoint3d XLine2Point { get; set; }
    [DataMember]
    public crawlPoint3d DimLinePoint { get; set; }
    [DataMember]
    public crawlPoint3d TextPosition { get; set; }

    [DataMember]
    public string DimensionText;
    [DataMember]
    public string DimensionStyleName;

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbRotatedDimension(RotatedDimension dim)
    {
        this.XLine1Point = new crawlPoint3d(dim.XLine1Point);
        this.XLine2Point = new crawlPoint3d(dim.XLine2Point);
        this.DimLinePoint = new crawlPoint3d(dim.DimLinePoint);
        this.TextPosition = new crawlPoint3d(dim.TextPosition);

        this.Layer = dim.Layer;
        this.Linetype = dim.Linetype;
        this.LineWeight = dim.LineWeight.ToString();
        this.Color = dim.Color.ToString();

        this.DimensionText = dim.DimensionText;
        this.DimensionStyleName = dim.DimensionStyleName;
    }
}

[DataContract]
public class crawlAcDbPoint3AngularDimension
{
    [DataMember]
    string ClassName = "AcDbPoint3AngularDimension";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d XLine1Point { get; set; }
    [DataMember]
    public crawlPoint3d XLine2Point { get; set; }
    [DataMember]
    public crawlPoint3d CenterPoint { get; set; }
    [DataMember]
    public crawlPoint3d TextPosition { get; set; }

    [DataMember]
    public string DimensionText;
    [DataMember]
    public string DimensionStyleName;

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbPoint3AngularDimension(Point3AngularDimension dim)
    {
        this.XLine1Point = new crawlPoint3d(dim.XLine1Point);
        this.XLine2Point = new crawlPoint3d(dim.XLine2Point);
        this.CenterPoint = new crawlPoint3d(dim.CenterPoint);
        this.TextPosition = new crawlPoint3d(dim.TextPosition);

        this.Layer = dim.Layer;
        this.Linetype = dim.Linetype;
        this.LineWeight = dim.LineWeight.ToString();
        this.Color = dim.Color.ToString();

        this.DimensionText = dim.DimensionText;
        this.DimensionStyleName = dim.DimensionStyleName;
    }
}

[DataContract]
public class crawlAcDbLineAngularDimension2
{
    [DataMember]
    string ClassName = "AcDbLineAngularDimension2";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d XLine1Start { get; set; }
    [DataMember]
    public crawlPoint3d XLine1End { get; set; }
    [DataMember]
    public crawlPoint3d XLine2Start { get; set; }
    [DataMember]
    public crawlPoint3d XLine2End { get; set; }
    [DataMember]
    public crawlPoint3d ArcPoint { get; set; }
    [DataMember]
    public crawlPoint3d TextPosition { get; set; }

    [DataMember]
    public string DimensionText;
    [DataMember]
    public string DimensionStyleName;

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbLineAngularDimension2(LineAngularDimension2 dim)
    {
        this.XLine1Start = new crawlPoint3d(dim.XLine1Start);
        this.XLine1End = new crawlPoint3d(dim.XLine1End);
        this.XLine2Start = new crawlPoint3d(dim.XLine2Start);
        this.XLine2End = new crawlPoint3d(dim.XLine2End);
        this.ArcPoint = new crawlPoint3d(dim.ArcPoint);
        this.TextPosition = new crawlPoint3d(dim.TextPosition);

        this.Layer = dim.Layer;
        this.Linetype = dim.Linetype;
        this.LineWeight = dim.LineWeight.ToString();
        this.Color = dim.Color.ToString();

        this.DimensionText = dim.DimensionText;
        this.DimensionStyleName = dim.DimensionStyleName;
    }
}

[DataContract]
public class crawlAcDbDiametricDimension
{
    [DataMember]
    string ClassName = "AcDbDiametricDimension";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d FarChordPoint { get; set; }
    [DataMember]
    public crawlPoint3d ChordPoint { get; set; }
    [DataMember]
    public crawlPoint3d TextPosition { get; set; }

    [DataMember]
    public string DimensionText;
    [DataMember]
    public string DimensionStyleName;

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbDiametricDimension(DiametricDimension dim)
    {
        this.FarChordPoint = new crawlPoint3d(dim.FarChordPoint);
        this.ChordPoint = new crawlPoint3d(dim.ChordPoint);
        this.TextPosition = new crawlPoint3d(dim.TextPosition);

        this.Layer = dim.Layer;
        this.Linetype = dim.Linetype;
        this.LineWeight = dim.LineWeight.ToString();
        this.Color = dim.Color.ToString();

        this.DimensionText = dim.DimensionText;
        this.DimensionStyleName = dim.DimensionStyleName;
    }
}

[DataContract]
public class crawlAcDbArcDimension
{
    [DataMember]
    string ClassName = "AcDbArcDimension";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d XLine1Point { get; set; }
    [DataMember]
    public crawlPoint3d XLine2Point { get; set; }
    [DataMember]
    public crawlPoint3d ArcPoint { get; set; }
    [DataMember]
    public crawlPoint3d TextPosition { get; set; }

    [DataMember]
    public string DimensionText;
    [DataMember]
    public string DimensionStyleName;

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbArcDimension(ArcDimension dim)
    {
        this.XLine1Point = new crawlPoint3d(dim.XLine1Point);
        this.XLine2Point = new crawlPoint3d(dim.XLine2Point);
        this.ArcPoint = new crawlPoint3d(dim.ArcPoint);
        this.TextPosition = new crawlPoint3d(dim.TextPosition);

        this.Layer = dim.Layer;
        this.Linetype = dim.Linetype;
        this.LineWeight = dim.LineWeight.ToString();
        this.Color = dim.Color.ToString();

        this.DimensionText = dim.DimensionText;
        this.DimensionStyleName = dim.DimensionStyleName;
    }
}

[DataContract]
public class crawlAcDbRadialDimension
{
    [DataMember]
    string ClassName = "AcDbRadialDimension";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d Center { get; set; }
    [DataMember]
    public crawlPoint3d ChordPoint { get; set; }
    [DataMember]
    public crawlPoint3d TextPosition { get; set; }

    [DataMember]
    public string DimensionText;
    [DataMember]
    public string DimensionStyleName;

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbRadialDimension(RadialDimension dim)
    {
        this.Center = new crawlPoint3d(dim.Center);
        this.ChordPoint = new crawlPoint3d(dim.ChordPoint);
        this.TextPosition = new crawlPoint3d(dim.TextPosition);

        this.Layer = dim.Layer;
        this.Linetype = dim.Linetype;
        this.LineWeight = dim.LineWeight.ToString();
        this.Color = dim.Color.ToString();

        this.DimensionText = dim.DimensionText;
        this.DimensionStyleName = dim.DimensionStyleName;
    }
}

[DataContract]
public class crawlAcDbHatch
{
    [DataMember]
    string ClassName = "AcDbHatch";
    [DataMember]
    string Color;

    [DataMember]
    double Area;
    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;
    [DataMember]
    string PatternName;

    [DataMember]
    List<crawlAcDbPolyline> Loops;

    public crawlAcDbHatch(Hatch hatch)
    {
        this.Area = hatch.Area;
        this.Layer = hatch.Layer;
        this.Linetype = hatch.Linetype;
        this.LineWeight = hatch.LineWeight.ToString();
        this.Color = hatch.Color.ToString();

        this.PatternName = hatch.PatternName;

        Loops = HatchToPolylines(hatch.ObjectId);
    }

    /// <summary>
    ///  Функция преобразования координат контура штриховки. Последовательно пробегает по каждому из контуров штриховки и преобразует их в полилинии
    /// </summary>
    /// <param name="hatchId">ObjectId штриховки Hatch</param>
    /// <returns>Список crawlAcDbPolyline - перечень контуров штриховки</returns>
    private List<crawlAcDbPolyline> HatchToPolylines(ObjectId hatchId)
    {
        List<crawlAcDbPolyline> result = new List<crawlAcDbPolyline>();

        //Исходный код для AutoCAD .Net
        //http://forums.autodesk.com/t5/NET/Restore-hatch-boundaries-if-they-have-been-lost-with-NET/m-p/3779514#M33429

        try
        {

            Hatch hatch = (Hatch)hatchId.GetObject(OpenMode.ForRead);
            if (hatch != null)
            {
                int nLoops = hatch.NumberOfLoops;
                for (int i = 0; i < nLoops; i++)
                {//Цикл по каждому из контуров штриховки
                    //Проверяем что контур является полилинией
                    HatchLoop loop = hatch.GetLoopAt(i);
                    if (loop.IsPolyline)
                    {
                        using (Polyline poly = new Polyline())
                        {
                            //Создаем полилинию из точек контура
                            int iVertex = 0;
                            foreach (BulgeVertex bv in loop.Polyline)
                            {
                                poly.AddVertexAt(iVertex++, bv.Vertex, bv.Bulge, 0.0, 0.0);
                            }
                            result.Add(new crawlAcDbPolyline(poly));
                        }
                    }
                    else
                    {//Если не удалось преобразовать контур к полилинии
                        //Выводим сообщение в командную строку
                        Crawl.cDebug.WriteLine("Ошибка обработки: Контур штриховки - не полилиния");
                        //Не будем брать исходный код для штриховок, контур который не сводится к полилинии
                    }
                }
            }
        }
        catch (Exception e)
        {
            Crawl.cDebug.WriteLine("Ошибка обработки штриховки: {0}", e.Message);
        }
        return result;
    }
}

[DataContract]
public class crawlAcDbSpline
{
    [DataMember]
    string ClassName = "AcDbSpline";
    [DataMember]
    string Color;

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;
    [DataMember]
    double Area;

    [DataMember]
    List<crawlPoint3d> Vertixes;

    public crawlAcDbSpline(Spline spline)
    {
        this.Area = spline.Area;

        this.Layer = spline.Layer;
        this.Linetype = spline.Linetype;
        this.LineWeight = spline.LineWeight.ToString();
        this.Color = spline.Color.ToString();

        Vertixes = getSplinePoints(spline);
    }


    private List<crawlPoint3d> getSplinePoints(Spline spline)
    {
        List<crawlPoint3d> result = new List<crawlPoint3d>();

        //Исходный пример из AutoCAD:
        //http://through-the-interface.typepad.com/through_the_interface/2007/04/iterating_throu.html
        //сильно в нем не разбирался, просто адаптирован.

        try
        {

            // Количество контрольных точек сплайна
            int vn = spline.NumControlPoints;

            //Цикл по всем контрольным точкам сплайна
            for (int i = 0; i < vn; i++)
            {
                // Could also get the 3D point here
                Point3d pt = spline.GetControlPointAt(i);

                result.Add(new crawlPoint3d(pt));
            }
        }
        catch
        {
            cDebug.WriteLine("Not a spline or something wrong");
        }
        return result;
    }
}

[DataContract]
public class crawlAcDbPoint
{
    [DataMember]
    string ClassName = "AcDbPoint";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d Position { get; set; }

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbPoint(DBPoint pnt)
    {
        this.Position = new crawlPoint3d(pnt.Position);

        this.Layer = pnt.Layer;
        this.Linetype = pnt.Linetype;
        this.LineWeight = pnt.LineWeight.ToString();
        this.Color = pnt.Color.ToString();
    }
}

[DataContract]
public class crawlAcDbBlockReference
{
    [DataMember]
    string ClassName = "AcDbBlockReference";
    [DataMember]
    string Color;

    [DataMember]
    public crawlPoint3d Position { get; set; }

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    [DataMember]
    string Name;

    [DataMember]
    List<crawlAcDbAttributeReference> Attributes;

    public crawlAcDbBlockReference(BlockReference blk)
    {
        this.Position = new crawlPoint3d(blk.Position);

        this.Layer = blk.Layer;
        this.Linetype = blk.Linetype;
        this.LineWeight = blk.LineWeight.ToString();
        this.Color = blk.Color.ToString();

        this.Name = blk.Name;

        Attributes = new List<crawlAcDbAttributeReference>();

        //http://through-the-interface.typepad.com/through_the_interface/2007/07/updating-a-spec.html
        foreach (ObjectId attId in blk.AttributeCollection)
        {
            AttributeReference attRef = (AttributeReference)attId.GetObject(OpenMode.ForRead);
            this.Attributes.Add(new crawlAcDbAttributeReference(attRef));
        }
    }
}

[DataContract]
public class crawlAcDbAttributeReference
{
    [DataMember]
    string ClassName = "AcDbAttributeReference";
    [DataMember]
    string Color;

    [DataMember]
    string Tag;
    [DataMember]
    string TextString;

    public crawlAcDbAttributeReference(AttributeReference attRef)
    {
        this.Tag = attRef.Tag;
        this.TextString = attRef.TextString;
        this.Color = attRef.Color.ToString();
    }
}

[DataContract]
public class crawlAcDbProxyEntity
{
    [DataMember]
    string ClassName = "AcDbProxyEntity";
    [DataMember]
    string Color;

    [DataMember]
    string Layer;
    [DataMember]
    string Linetype;
    [DataMember]
    string LineWeight;

    public crawlAcDbProxyEntity(ProxyEntity prxy)
    {
        this.Layer = prxy.Layer;
        this.Linetype = prxy.Linetype;
        this.LineWeight = prxy.LineWeight.ToString();
        this.Color = prxy.Color.ToString();
    }
}

[DataContract]
public class crawlAcDbBlockTableRecord
{
    [DataMember]
    string ClassName = "AcDbBlockTableRecord";
    [DataMember]
    string Name;
    [DataMember]
    string FilePath;

    public crawlAcDbBlockTableRecord(BlockTableRecord btr, string filePath)
    {
        this.Name = btr.Name;
        this.FilePath = filePath;
    }

}
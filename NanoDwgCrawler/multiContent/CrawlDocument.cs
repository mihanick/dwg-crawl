using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.Serialization;
using Teigha.DatabaseServices;
using Crawl;
using HostMgd.ApplicationServices;

using TeighaApp = HostMgd.ApplicationServices.Application;
using HostMgd.EditorInput;


namespace Crawl
{
    [DataContract]
    class CrawlDocument
    {
        [DataMember]
        string Path;
        [DataMember]
        int id;

        Document teighaDocument;

        public SqlDb sqlDB;

        public CrawlDocument(string dwgPath)
        {
            try
            {
                sqlDB = new SqlDb();

                this.Path = dwgPath;

                Document aDoc = TeighaApp.DocumentManager.Open(dwgPath);
                this.teighaDocument = aDoc;
                this.id = teighaDocument.GetHashCode();

                string docJson = jsonHelper.To<CrawlDocument>(this);

                newDocument(aDoc, docJson, dwgPath);

                aDoc.CloseAndDiscard();
            }
            catch
            {
                cDebug.WriteLine("Не могу просканировать файл {0}", dwgPath);
            }
        }

        private void newDocument(Document aDoc, string docJson, string dwgPath)
        {
            int docHash = aDoc.GetHashCode();

            this.sqlDB.InsertIntoFiles(dwgPath, docHash, docJson);

            using (Transaction tr = aDoc.TransactionManager.StartTransaction())
            {
                PromptSelectionResult r = aDoc.Editor.SelectAll();

                foreach (SelectedObject obj in r.Value)
                {
                    string objId = obj.ObjectId.ToString();
                    string objectJson = jsonGetObjectData(obj.ObjectId);
                    string objectClass = obj.ObjectId.ObjectClass.Name;

                    this.sqlDB.SaveObjectData(objId, docHash, objectJson, objectClass);
                }
            }
        }

        private void newDocument(ObjectId objId, string docJson)
        {
            //http://www.theswamp.org/index.php?topic=37860.0
            Document aDoc = Application.DocumentManager.GetDocument(objId.Database);

            if (objId.ObjectClass.Name == "AcDbBlockReference")
            {
                BlockReference blk = (BlockReference)objId.GetObject(OpenMode.ForRead);
                BlockTableRecord btr = (BlockTableRecord)blk.BlockTableRecord.GetObject(OpenMode.ForRead);

                int docHash = btr.GetHashCode();

                this.sqlDB.InsertIntoFiles("", docHash, docJson);

                using (Transaction tr = aDoc.TransactionManager.StartTransaction())
                {
                    foreach (ObjectId obj in btr)
                    {
                        string objectJson = jsonGetObjectData(obj);
                        string objectClass = obj.ObjectClass.Name;

                        this.sqlDB.SaveObjectData(obj.ToString(), docHash, objectJson, objectClass);
                    }
                }
            }
            else if (objId.ObjectClass.Name == "AcDbProxyEntity")
            {
                Entity ent = (Entity)objId.GetObject(OpenMode.ForRead);
                DBObjectCollection dbo = new DBObjectCollection();
                ent.Explode(dbo);

                throw new NotImplementedException();

            }
        }


        private string jsonGetObjectData(ObjectId id_platf)
        {
            string result = "";

            try
            {//Всякое может случиться
                //Открываем переданный в функцию объект на чтение, преобразуем его к Entity
                Entity ent = (Entity)id_platf.GetObject(OpenMode.ForWrite);

                //Далее последовательно проверяем класс объекта на соответствие классам основных примитивов

                if (id_platf.ObjectClass.Name == "AcDbLine")
                {//Если объект - отрезок (line)
                    crawlAcDbLine kline = new crawlAcDbLine((Line)ent); //Преобразуем к типу линия
                    result = jsonHelper.To<crawlAcDbLine>(kline);
                }
                else if (id_platf.ObjectClass.Name == "AcDbPolyline")
                {//Если объект - полилиния
                    Polyline kpLine = (Polyline)ent;
                    crawlAcDbPolyline jpline = new crawlAcDbPolyline(kpLine);
                    result = jsonHelper.To<crawlAcDbPolyline>(jpline);
                }
                else if (id_platf.ObjectClass.Name == "AcDb2dPolyline")
                {//2D полилиния - такие тоже попадаются
                    Polyline2d kpLine = (Polyline2d)ent;
                    crawlAcDbPolyline jpline = new crawlAcDbPolyline(kpLine);
                    result = jsonHelper.To<crawlAcDbPolyline>(jpline);
                }
                else if (id_platf.ObjectClass.Name == "AcDb3dPolyline")
                {//2D полилиния - такие тоже попадаются
                    Polyline3d kpLine = (Polyline3d)ent;

                    crawlAcDbPolyline jpline = new crawlAcDbPolyline(kpLine);
                    result = jsonHelper.To<crawlAcDbPolyline>(jpline);
                }
                else if (id_platf.ObjectClass.Name == "AcDbText")
                { //Текст
                    DBText dbtxt = (DBText)ent;
                    crawlAcDbText jtext = new crawlAcDbText(dbtxt);
                    result = jsonHelper.To<crawlAcDbText>(jtext);
                }
                else if (id_platf.ObjectClass.Name == "AcDbMText")
                {//Мтекст
                    MText mtxt = (MText)ent;
                    crawlAcDbMText jtext = new crawlAcDbMText(mtxt);
                    result = jsonHelper.To<crawlAcDbMText>(jtext);
                }
                else if (id_platf.ObjectClass.Name == "AcDbArc")
                {//Дуга
                    Arc arc = (Arc)ent;
                    crawlAcDbArc cArc = new crawlAcDbArc(arc);
                    result = jsonHelper.To<crawlAcDbArc>(cArc);
                }
                else if (id_platf.ObjectClass.Name == "AcDbCircle")
                {//Окружность
                    Circle circle = (Circle)ent;
                    crawlAcDbCircle cCircle = new crawlAcDbCircle(circle);
                    result = jsonHelper.To<crawlAcDbCircle>(cCircle);
                }
                else if (id_platf.ObjectClass.Name == "AcDbEllipse")
                {  //Эллипс
                    Ellipse el = (Ellipse)ent;
                    crawlAcDbEllipse cEll = new crawlAcDbEllipse(el);
                    result = jsonHelper.To<crawlAcDbEllipse>(cEll);
                }
                else if (id_platf.ObjectClass.Name == "AcDbRotatedDimension")
                {//Размер повернутый
                    RotatedDimension dim = (RotatedDimension)ent;

                    crawlAcDbRotatedDimension rDim = new crawlAcDbRotatedDimension(dim);
                    result = jsonHelper.To<crawlAcDbRotatedDimension>(rDim);
                }

                else if (id_platf.ObjectClass.Name == "AcDbPoint3AngularDimension")
                {//Угловой размер по 3 точкам
                    Point3AngularDimension dim = (Point3AngularDimension)ent;

                    crawlAcDbPoint3AngularDimension rDim = new crawlAcDbPoint3AngularDimension(dim);
                    result = jsonHelper.To<crawlAcDbPoint3AngularDimension>(rDim);
                }

                else if (id_platf.ObjectClass.Name == "AcDbLineAngularDimension2")
                {//Еще угловой размер по точкам
                    LineAngularDimension2 dim = (LineAngularDimension2)ent;

                    crawlAcDbLineAngularDimension2 rDim = new crawlAcDbLineAngularDimension2(dim);
                    result = jsonHelper.To<crawlAcDbLineAngularDimension2>(rDim);
                }
                else if (id_platf.ObjectClass.Name == "AcDbDiametricDimension")
                {  //Размер диаметра окружности
                    DiametricDimension dim = (DiametricDimension)ent;
                    crawlAcDbDiametricDimension rDim = new crawlAcDbDiametricDimension(dim);
                    result = jsonHelper.To<crawlAcDbDiametricDimension>(rDim);
                }
                else if (id_platf.ObjectClass.Name == "AcDbArcDimension")
                {  //Дуговой размер
                    ArcDimension dim = (ArcDimension)ent;
                    crawlAcDbArcDimension rDim = new crawlAcDbArcDimension(dim);
                    result = jsonHelper.To<crawlAcDbArcDimension>(rDim);

                }
                else if (id_platf.ObjectClass.Name == "AcDbRadialDimension")
                {  //Радиальный размер
                    RadialDimension dim = (RadialDimension)ent;
                    crawlAcDbRadialDimension rDim = new crawlAcDbRadialDimension(dim);
                    result = jsonHelper.To<crawlAcDbRadialDimension>(rDim);
                }
                else if (id_platf.ObjectClass.Name == "AcDbAttributeDefinition")
                {  //Атрибут блока
                    AttributeDefinition ad = (AttributeDefinition)ent;

                    crawlAcDbAttributeDefinition atd = new crawlAcDbAttributeDefinition(ad);
                    result = jsonHelper.To<crawlAcDbAttributeDefinition>(atd);
                }
                else if (id_platf.ObjectClass.Name == "AcDbHatch")
                {//Штриховка
                    Teigha.DatabaseServices.Hatch htch = ent as Teigha.DatabaseServices.Hatch;

                    crawlAcDbHatch cHtch = new crawlAcDbHatch(htch);
                    result = jsonHelper.To<crawlAcDbHatch>(cHtch);
                }
                else if (id_platf.ObjectClass.Name == "AcDbSpline")
                {//Сплайн
                    Spline spl = ent as Spline;

                    crawlAcDbSpline cScpline = new crawlAcDbSpline(spl);
                    result = jsonHelper.To<crawlAcDbSpline>(cScpline);
                }
                else if (id_platf.ObjectClass.Name == "AcDbPoint")
                {//Точка
                    DBPoint Pnt = ent as DBPoint;
                    crawlAcDbPoint pt = new crawlAcDbPoint(Pnt);
                    result = jsonHelper.To<crawlAcDbPoint>(pt);
                }

                else if (id_platf.ObjectClass.Name == "AcDbBlockReference")
                {//Блок
                    BlockReference blk = ent as BlockReference;
                    crawlAcDbBlockReference cBlk = new crawlAcDbBlockReference(blk);

                    result = jsonHelper.To<crawlAcDbBlockReference>(cBlk);

                    newDocument(id_platf, result);
                }
                else if (id_platf.ObjectClass.Name == "AcDbProxyEntity")
                {//Блок
                    ProxyEntity pxy = ent as ProxyEntity;


                    crawlAcDbProxyEntity cBlk = new crawlAcDbProxyEntity(pxy);

                    result = jsonHelper.To<crawlAcDbProxyEntity>(cBlk);

                    newDocument(id_platf, result);
                }
                /*


            else if (id_platf.ObjectClass.Name == "AcDbLeader")
            {  //Выноска Autocad
                Leader ld = (Leader)ent;

                if (ld.EndPoint.Z != 0 || ld.StartPoint.Z != 0)
                {
                    //ed.WriteMessage("DEBUG: Преобразован объект: Выноска Autocad");

                    ld.EndPoint = new Point3d(ld.EndPoint.X, ld.EndPoint.Y, 0);
                    ld.StartPoint = new Point3d(ld.StartPoint.X, ld.StartPoint.Y, 0);

                    result = true;
                };

            }
            /*
        else if (id_platf.ObjectClass.Name == "AcDbPolygonMesh")
        {
             BUG: В платформе нет API для доступа к вершинам сетей AcDbPolygonMesh и AcDbPolygonMesh и AcDbSurface
                     
        }
        else if (id_platf.ObjectClass.Name == "AcDbSolid")
        {
             BUG: Чтобы плющить Solid-ы нужны API функции 3d
        }
        else if (id_platf.ObjectClass.Name == "AcDbRegion")
        {
            Region rgn = ent as Region;
            BUG: нет свойств у региона
        }
                
        */
                else
                {
                    //Если объект не входит в число перечисленных типов,
                    //то выводим в командную строку класс этого необработанного объекта

                    cDebug.WriteLine("Не могу обработать тип объекта: " + id_platf.ObjectClass.Name);
                }
            }
            catch (System.Exception ex)
            {
                //Если что-то сломалось, то в командную строку выводится ошибка
                cDebug.WriteLine("Не могу преобразовать - ошибка: " + ex.Message);
            };

            //Возвращаем значение функции
            return result;
        }


    }
}

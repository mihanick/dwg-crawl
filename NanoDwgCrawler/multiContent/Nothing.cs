using Teigha.DatabaseServices;
using Teigha.Geometry;

// Не работает вместе c Multicad.Runtime
using PlatformDb = Teigha;

//Использование определенных типов, которые определены и в платформе и в мультикаде
using Hatch = Teigha.DatabaseServices.Hatch;
using Point3d = Teigha.Geometry.Point3d;
using Polyline3d = Teigha.DatabaseServices.Polyline3d;



namespace ogpUtils.Content
{
    public partial class Commands
    {

        public bool FlattenByPlatform(ObjectId id_platf, bool explodeBlocks)
        {
            bool result = false; //Пока ничего не плющили результат равен false

            try
            {//Всякое может случиться
                //Открываем переданный в функцию объект на чтение, преобразуем его к Entity
                Entity ent = (Entity)id_platf.GetObject(OpenMode.ForWrite);

                //Далее последовательно проверяем класс объекта на соответствие классам основных
                //примитивов

                if (id_platf.ObjectClass.Name == "AcDbLine")
                {//Если объект - отрезок (line)
                    Line kline = ent as Line; //Преобразуем к типу линия



                    if (kline.StartPoint.Z != 0 || kline.EndPoint.Z != 0 || kline.Thickness != 0) //если начальная или конечная точка не лежит на ХоУ
                    {

                        //То зануляем координату Z начала или конца
                        kline.StartPoint = new Teigha.Geometry.Point3d(kline.StartPoint.X, kline.StartPoint.Y, 0);
                        kline.EndPoint = new Teigha.Geometry.Point3d(kline.EndPoint.X, kline.EndPoint.Y, 0);

                        kline.Thickness = 0;

                        //ed.WriteMessage("DEBUG: Преобразован объект: линия");

                        //Возвращаем результат
                        result = true;
                    };

                }
                else if (id_platf.ObjectClass.Name == "AcDbBlockReference")
                {//Блок
                    BlockReference blk = ent as BlockReference;

                    //Если блок просто размещен в пронстранстве (не на XoY)
                    if (blk.Position.Z != 0)
                    {
                        //То просто зануляем ему координату Z
                        blk.Position = new Point3d(blk.Position.X, blk.Position.Y, 0);
                        //ed.WriteMessage("DEBUG: Преобразован объект: блок(перемещен)");

                        //Увеличиваем число перемещенных блоков
                        result = true;
                    };
                    //Если найден блок - его разбираем, и если в нем есть неплоские объекты - плющим
                    if (ExplodeBlock(id_platf, explodeBlocks))
                    {
                        //ed.WriteMessage("DEBUG: Преобразован объект: блок (взорван)");

                        //Для отчетности увеличиваем счетчик взорванных блоков
                        result = true;
                    }
                }
                else if (id_platf.ObjectClass.Name == "AcDbProxyEntity")
                {//Блок
                    ProxyEntity pxy = ent as ProxyEntity;

                    //Если найден Прокси - его разбираем, и если в нем есть неплоские объекты - плющим
                    if (ExplodeProxy(id_platf, explodeBlocks))
                    {
                        //ed.WriteMessage("DEBUG: Преобразован объект: блок (взорван)");

                        //Для отчетности увеличиваем счетчик взорванных блоков
                        result = true;
                    }
                }
                else if (id_platf.ObjectClass.Name == "AcDbPolyline")
                {//Если объект - полилиния
                    Polyline kpLine = (Polyline)ent;

                    //Если у полинии Elevation не 0 или полилинию можно расплющить (там идет проверка по каждой вершине)
                    if (kpLine.Elevation != 0 || kpLine.Thickness != 0 || flattenPolyline(ent))
                    {
                        kpLine.Thickness = 0;
                        kpLine.Elevation = 0;

                        //ed.WriteMessage("DEBUG: Преобразован объект: полилиния");
                        result = true;
                    };
                }
                else if (id_platf.ObjectClass.Name == "AcDb3dPolyline")
                {//2D полилиния - такие тоже попадаются
                    Polyline3d kpLine = (Polyline3d)ent;

                    if (flattenPolyline(ent))
                    {
                        //ed.WriteMessage("DEBUG: Преобразован объект: 3d полилиния");
                        result = true;
                    };
                }
                else if (id_platf.ObjectClass.Name == "AcDb2dPolyline")
                {//2D полилиния - такие тоже попадаются
                    Polyline2d kpLine = (Polyline2d)ent;

                    if (kpLine.Elevation != 0 || kpLine.Thickness != 0 || flattenPolyline(ent))
                    {
                        kpLine.Thickness = 0;
                        kpLine.Elevation = 0;

                        //ed.WriteMessage("DEBUG: Преобразован объект: 2d полилиния");
                        result = true;
                    };
                }
                else if (id_platf.ObjectClass.Name == "AcDbCircle")
                {//Окружность
                    Circle cir = (Circle)ent;
                    if (cir.Center.Z != 0 || cir.StartPoint.Z != 0 || cir.EndPoint.Z != 0 || cir.Thickness != 0)
                    {
                        cir.Center = new Teigha.Geometry.Point3d(cir.Center.X, cir.Center.Y, 0);
                        cir.Thickness = 0;
                        //TODO: Проецирование окружностей в эллипс

                        //ed.WriteMessage("DEBUG: Преобразован объект: окружность");
                        result = true;
                    }
                }
                else if (id_platf.ObjectClass.Name == "AcDbArc")
                {//Дуга
                    Arc arc = ent as Arc;

                    //if (arc.Center.Z != 0 || arc.StartPoint.Z != 0 || arc.EndPoint.Z != 0)
                    if (arc.Center.Z != 0 || arc.Thickness != 0)
                    {
                        arc.Center = new Point3d(arc.Center.X, arc.Center.Y, 0);
                        arc.Thickness = 0;
                        //BUG: StartPoint и EndPoint дуги по-моему только READ-ONLY
                        //arc.StartPoint = new Point3d(arc.StartPoint.X, arc.StartPoint.Y, 0);
                        //arc.EndPoint = new Point3d(arc.EndPoint.X, arc.EndPoint.Y, 0);

                        //ed.WriteMessage("DEBUG: Преобразован объект: Дуга");
                        result = true;
                    }
                }
                else if (id_platf.ObjectClass.Name == "AcDbPoint")
                {//Точка
                    DBPoint Pnt = ent as DBPoint;
                    if (Pnt.Position.Z != 0)
                    {
                        Pnt.Position = new Teigha.Geometry.Point3d(Pnt.Position.X, Pnt.Position.Y, 0);

                        //ed.WriteMessage("DEBUG: Преобразован объект: Точка");
                        result = true;
                    }
                }
                else if (id_platf.ObjectClass.Name == "AcDbSpline")
                {//Сплайн
                    Spline spl = ent as Spline;

                    //Если сплайн можно расплющить (за это отвечает отдельная функция)
                    if (flattenSpline(ent))
                    {
                        //ed.WriteMessage("DEBUG: Преобразован объект: Сплайн");
                        result = true;
                    }
                }
                else if (id_platf.ObjectClass.Name == "AcDbText")
                { //Текст
                    DBText dbtxt = (DBText)ent;
                    if (dbtxt.Position.Z != 0 || dbtxt.Thickness != 0)
                    {
                        dbtxt.Thickness = 0;
                        dbtxt.Position = new Teigha.Geometry.Point3d(dbtxt.Position.X, dbtxt.Position.Y, 0);
                        //ed.WriteMessage("DEBUG: Преобразован объект: Текст");
                        result = true;
                    }
                }
                else if (id_platf.ObjectClass.Name == "AcDbMText")
                {//Мтекст
                    MText mtxt = (MText)ent;
                    if (mtxt.Location.Z != 0)
                    {
                        mtxt.Location = new Teigha.Geometry.Point3d(mtxt.Location.X, mtxt.Location.Y, 0);
                        //ed.WriteMessage("DEBUG: Преобразован объект: Мтекст");
                        result = true;
                    };
                }
                else if (id_platf.ObjectClass.Name == "AcDbHatch")
                {//Штриховка
                    Teigha.DatabaseServices.Hatch htch = ent as Teigha.DatabaseServices.Hatch;

                    //Если штриховка имеет Elevation не 0 или ее можно расплющить
                    if (htch.Elevation != 0 || FlattenHatch(id_platf))
                    {
                        htch.Elevation = 0;


                        //ed.WriteMessage("DEBUG: Преобразован объект: Штриховка");
                        result = true;
                    }
                }
                else if (id_platf.ObjectClass.Name == "AcDbRotatedDimension")
                {//Размер повернутый
                    RotatedDimension dim = (RotatedDimension)ent;

                    //Проверяем, имеют ли задающие точки размера ненулевую координату Z
                    if (dim.XLine1Point.Z != 0 || dim.XLine2Point.Z != 0 || dim.DimLinePoint.Z != 0 || dim.TextPosition.Z != 0)
                    {
                        dim.XLine1Point = new Point3d(dim.XLine1Point.X, dim.XLine1Point.Y, 0);
                        dim.XLine2Point = new Point3d(dim.XLine2Point.X, dim.XLine2Point.Y, 0);
                        dim.DimLinePoint = new Point3d(dim.DimLinePoint.X, dim.DimLinePoint.Y, 0);
                        dim.TextPosition = new Point3d(dim.TextPosition.X, dim.TextPosition.Y, 0);

                        //ed.WriteMessage("DEBUG: Преобразован объект: повернутый размер");

                        result = true;
                    };
                }
                else if (id_platf.ObjectClass.Name == "AcDbPoint3AngularDimension")
                {//Угловой размер по 3 точкам
                    Point3AngularDimension dim = (Point3AngularDimension)ent;
                    if (dim.XLine1Point.Z != 0 || dim.XLine2Point.Z != 0 || dim.CenterPoint.Z != 0 || dim.TextPosition.Z != 0)
                    {

                        dim.XLine1Point = new Point3d(dim.XLine1Point.X, dim.XLine1Point.Y, 0);
                        dim.XLine2Point = new Point3d(dim.XLine2Point.X, dim.XLine2Point.Y, 0);
                        dim.CenterPoint = new Point3d(dim.CenterPoint.X, dim.CenterPoint.Y, 0);

                        dim.TextPosition = new Point3d(dim.TextPosition.X, dim.TextPosition.Y, 0);

                        //ed.WriteMessage("DEBUG: Преобразован объект: Угловой размер по трем точкам");

                        result = true;
                    };
                }
                else if (id_platf.ObjectClass.Name == "AcDbLineAngularDimension2")
                {//Еще угловой размер по точкам
                    LineAngularDimension2 dim = (LineAngularDimension2)ent;

                    if (dim.XLine1Start.Z != 0 || dim.XLine1End.Z != 0 || dim.XLine1Start.Z != 0 || dim.XLine2End.Z != 0 || dim.ArcPoint.Z != 0 || dim.TextPosition.Z != 0)
                    {

                        dim.XLine1Start = new Point3d(dim.XLine1Start.X, dim.XLine1Start.Y, 0);
                        dim.XLine1End = new Point3d(dim.XLine1End.X, dim.XLine1End.Y, 0);
                        dim.XLine2Start = new Point3d(dim.XLine2Start.X, dim.XLine2Start.Y, 0);
                        dim.XLine2End = new Point3d(dim.XLine2End.X, dim.XLine2End.Y, 0);
                        dim.ArcPoint = new Point3d(dim.ArcPoint.X, dim.ArcPoint.Y, 0);

                        dim.TextPosition = new Point3d(dim.TextPosition.X, dim.TextPosition.Y, 0);

                        //ed.WriteMessage("DEBUG: Преобразован объект: Угловой размер по 5 точкам");

                        result = true;
                    };
                }
                else if (id_platf.ObjectClass.Name == "AcDbDiametricDimension")
                {  //Размер диаметра окружности
                    DiametricDimension dim = (DiametricDimension)ent;

                    if (dim.FarChordPoint.Z != 0 || dim.ChordPoint.Z != 0 || dim.TextPosition.Z != 0)
                    {
                        dim.FarChordPoint = new Point3d(dim.FarChordPoint.X, dim.FarChordPoint.Y, 0);
                        dim.ChordPoint = new Point3d(dim.ChordPoint.X, dim.ChordPoint.Y, 0);
                        dim.TextPosition = new Point3d(dim.TextPosition.X, dim.TextPosition.Y, 0);

                        //ed.WriteMessage("DEBUG: Преобразован объект: Диаметральный размер");

                        result = true;
                    };
                }
                else if (id_platf.ObjectClass.Name == "AcDbArcDimension")
                {  //Дуговой размер
                    ArcDimension dim = (ArcDimension)ent;

                    if (dim.XLine1Point.Z != 0 || dim.XLine2Point.Z != 0 || dim.ArcPoint.Z != 0 || dim.TextPosition.Z != 0)
                    {
                        dim.XLine1Point = new Point3d(dim.XLine1Point.X, dim.XLine1Point.Y, 0);
                        dim.XLine2Point = new Point3d(dim.XLine2Point.X, dim.XLine2Point.Y, 0);
                        dim.ArcPoint = new Point3d(dim.ArcPoint.X, dim.ArcPoint.Y, 0);
                        dim.TextPosition = new Point3d(dim.TextPosition.X, dim.TextPosition.Y, 0);

                        //ed.WriteMessage("DEBUG: Преобразован объект: Дуговой размер");

                        result = true;
                    };

                }
                else if (id_platf.ObjectClass.Name == "AcDbRadialDimension")
                {  //Радиальный размер
                    RadialDimension dim = (RadialDimension)ent;

                    if (dim.Center.Z != 0 || dim.ChordPoint.Z != 0 || dim.TextPosition.Z != 0)
                    {
                        dim.Center = new Point3d(dim.Center.X, dim.Center.Y, 0);
                        dim.ChordPoint = new Point3d(dim.ChordPoint.X, dim.ChordPoint.Y, 0);
                        dim.TextPosition = new Point3d(dim.TextPosition.X, dim.TextPosition.Y, 0);

                        //ed.WriteMessage("DEBUG: Преобразован объект: Радиальный размер");

                        result = true;
                    };

                }
                else if (id_platf.ObjectClass.Name == "AcDbEllipse")
                {  //Эллипс
                    Ellipse el = (Ellipse)ent;

                    /*if (el.Center.Z != 0 || el.StartPoint.Z != 0 || el.EndPoint.Z != 0)*/

                    if (el.Center.Z != 0)
                    {

                        el.Center = new Point3d(el.Center.X, el.Center.Y, 0);
                        /*
                        BUG: не поддерживается платформой перемещение начальной и конечной точек эллипса
                        el.StartPoint = new Point3d(el.StartPoint.X, el.StartPoint.Y, 0);
                        el.EndPoint = new Point3d(el.EndPoint.X, el.EndPoint.Y, 0);
                        */

                        //ed.WriteMessage("DEBUG: Преобразован объект: Эллипс");

                        result = true;
                    };

                }
                else if (id_platf.ObjectClass.Name == "AcDbAttributeDefinition")
                {  //Атрибут блока
                    AttributeDefinition ad = (AttributeDefinition)ent;

                    if (ad.Position.Z != 0)
                    {
                        //ed.WriteMessage("DEBUG: Преобразован объект: Атрибут блока");
                        ad.Position = new Point3d(ad.Position.X, ad.Position.Y, 0);

                        result = true;
                    };
                }

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

                    ed.WriteMessage("Не могу обработать тип объекта: " + id_platf.ObjectClass.Name);
                }
            }
            catch (PlatformDb.Runtime.Exception ex)
            {
                //Если что-то сломалось, то в командную строку выводится ошибка
                ed.WriteMessage("Не могу преобразовать - ошибка: " + ex.Message);
            };

            //Возвращаем значение функции
            return result;
        }

        private bool flattenSpline(Entity ent)
        {
            /*
            Функция плющит сплайн, последовательно зануляя координату Z каждой из его вершин
            На входе принимает Entity
            выдает true, если какая-либо из контрольных точек имела ненулевую кординату Z
            */
            bool result = false;

            //Исходный пример из AutoCAD:
            //http://through-the-interface.typepad.com/through_the_interface/2007/04/iterating_throu.html
            //сильно в нем не разбирался, просто адаптирован.

            try
            {
                Spline spl = (Spline)ent;
                if (spl != null)
                {
                    // Количество контрольных точек сплайна
                    int vn = spl.NumControlPoints;

                    //Цикл по всем контрольным точкам сплайна
                    for (int i = 0; i < vn; i++)
                    {
                        // Could also get the 3D point here
                        Point3d pt = spl.GetControlPointAt(i);
                        if (pt.Z != 0)
                        {
                            spl.SetControlPointAt(i, new Point3d(pt.X, pt.Y, 0));
                            result = true;
                        }
                    }
                }
            }
            catch
            {
                //Not a spline or something wrong
            }
            return result;
        }

        public bool flattenPolyline(Entity entline)
        {
            /*
             * Функция принимает на входе Entity и возвращает true, если эта
             * Entity является простой полилинией или 2dPolyline
             * И хотя бы одна из вершин этой полилинии имеет ненулевую координату Z 
            */
            bool result = false;

            //Исходный пример из AutoCAD .Net
            //http://through-the-interface.typepad.com/through_the_interface/2007/04/iterating_throu.html


            //Transaction tr = acCurDb.TransactionManager.StartTransaction();
            //1.10.2013 Siralex: Транзакция не нужна, т.к. команда стартует ее
            // вместо tr.getObject(id, OpenMode), можно пользоваться id.getObject(OpenMode);

            // If a "lightweight" (or optimized) polyline
            Polyline lwp = entline as Polyline;

            if (lwp != null)
            {
                // Use a for loop to get each vertex, one by one
                int vn = lwp.NumberOfVertices;
                for (int i = 0; i < vn; i++)
                {
                    // Could also get the 3D point here
                    Point3d pt = lwp.GetPoint3dAt(i);
                    if (pt.Z != 0)
                    {
                        //Назначаем новую вершину полилинии
                        lwp.SetPointAt(i, new Point2d(pt.X, pt.Y));
                        result = true;
                    }
                }
            }
            else
            {
                // If an old-style, 2D polyline
                Polyline2d p2d = entline as Polyline2d;
                if (p2d != null)
                {
                    // Use foreach to get each contained vertex
                    foreach (ObjectId vId in p2d)
                    {
                        Vertex2d v2d =
                          (Vertex2d)
                            vId.GetObject(
                            OpenMode.ForWrite
                          );
                        if (v2d.Position.Z != 0)
                        {
                            v2d.Position = new Point3d(v2d.Position.X, v2d.Position.Y, 0);
                            result = true;
                        };
                    }
                }
                else
                {
                    // If an old-style, 3D polyline
                    Polyline3d p3d = entline as Polyline3d;
                    if (p3d != null)
                    {
                        // Use foreach to get each contained vertex
                        foreach (ObjectId vId in p3d)
                        {
                            PolylineVertex3d v3d =
                              (PolylineVertex3d)
                                vId.GetObject(OpenMode.ForWrite);
                            if (v3d.Position.Z != 0)
                            {
                                v3d.Position = new Point3d(v3d.Position.X, v3d.Position.Y, 0);
                                result = true;
                            };
                        }
                    }
                }
            }
            return result;
        }

        private bool ExplodeBlock(ObjectId BlockId, bool eraseOrig)
        {
            /*
             * Функция взрывает блок и плющит то что осталось от взрыва
             * Принимает на входе ObjectId блока
             * И eraseOrig переменную, которая управляет удалением исходного блока
             * 
             */

            //Считаем что пока в блоке нечего плющить:
            bool SomethingToFlatten = false;


            // Это коллекция объектов, которая будет включать все элементы взорванного блока
            DBObjectCollection objs = new DBObjectCollection();

            //Открываем на чтение блок
            Entity ent =
              (Entity)BlockId.GetObject(
                OpenMode.ForRead
              );

            // Взрываем блок в нашу коллекцию объектов
            ent.Explode(objs);

            // Открываем текущее пространство на запись 
            BlockTableRecord btr =
              (BlockTableRecord)acCurDb.CurrentSpaceId.GetObject(
                OpenMode.ForWrite
              );

            //Открываем транзакцию
            Transaction tr =
              acCurDb.TransactionManager.StartTransaction();
            using (tr)
            {
                // Пробегаем по коллекции объектов и 
                // каждый из них добавляем к текущему пространству
                foreach (DBObject obj in objs)
                {
                    //преобразуем объект к Entity
                    Entity entExplode = (Entity)obj;
                    //Добавляем эту Entity в пространство
                    btr.AppendEntity(entExplode);
                    //Добавляем к транзакции новые объекты
                    tr.AddNewlyCreatedDBObject(entExplode, true);

                    //Проверяем, есть ли в составе блока объекты, 
                    //которые нужно расплющить (и в этом случае все входящие блоки плющим)
                    //Покольку только исходный блок нужно оставить, а все рекурсивно входящие в него
                    //подблоки в этом случае можно удалить.
                    if (FlattenByPlatform(entExplode.ObjectId, true))
                    {
                        //Здесь получается рекурсивный вызов плющилки с принудительным взрывом блоков
                        SomethingToFlatten = true;
                    }
                };

                //Если блок плоский, то и нечего его взрывать
                if (!SomethingToFlatten)
                {
                    //Удаляем что мы навзрывали - объекты не нужны на чертеже
                    foreach (DBObject obj in objs)
                    {
                        Entity entExplode = obj as Entity;
                        entExplode.Erase();
                    }
                    //Соответственно, если были неплоские примитивы, то результаты взрыва и расплющивания блока
                    //остаются на чертеже
                }
                else if (eraseOrig)
                //Проверим, если нужно - удалим исходный блок
                {
                    ent.UpgradeOpen();//открываем блок на запись
                    //и удаляем
                    ent.Erase();
                };

                // And then we commit
                tr.Commit();

                //Возвращаем значение (было что плющить)
                return SomethingToFlatten;
            }
        }

        private bool ExplodeProxy(ObjectId ProxyId, bool eraseOrig)
        {
            /*
             * Функция взрывает блок и плющит то что осталось от взрыва
             * Принимает на входе ObjectId блока
             * И eraseOrig переменную, которая управляет удалением исходного блока
             * 
             */

            //Считаем что пока в блоке нечего плющить:
            bool SomethingToFlatten = false;

            // Это коллекция объектов, которая будет включать все элементы взорванного блока
            DBObjectCollection objs = new DBObjectCollection();

            //Открываем на чтение блок
            Entity ent =
              (Entity)ProxyId.GetObject(
                OpenMode.ForRead
              );

            // Взрываем блок в нашу коллекцию объектов
            ent.Explode(objs);

            // Открываем текущее пространство на запись 
            BlockTableRecord btr =
              (BlockTableRecord)acCurDb.CurrentSpaceId.GetObject(
                OpenMode.ForWrite
              );

            //Открываем транзакцию
            Transaction tr =
              acCurDb.TransactionManager.StartTransaction();
            using (tr)
            {
                // Пробегаем по коллекции объектов и 
                // каждый из них добавляем к текущему пространству
                foreach (DBObject obj in objs)
                {
                    //преобразуем объект к Entity
                    Entity entExplode = (Entity)obj;
                    //Добавляем эту Entity в пространство
                    btr.AppendEntity(entExplode);
                    //Добавляем к транзакции новые объекты
                    tr.AddNewlyCreatedDBObject(entExplode, true);

                    //Проверяем, есть ли в составе блока объекты, 
                    //которые нужно расплющить (и в этом случае все входящие блоки плющим)
                    //Поскольку только исходный блок нужно оставить, а все рекурсивно входящие в него
                    //подблоки в этом случае можно удалить.
                    if (FlattenByPlatform(entExplode.ObjectId, true))
                    {
                        //Здесь получается рекурсивный вызов плющилки с принудительным взрывом блоков
                        SomethingToFlatten = true;
                    }
                };
                //Если блок плоский, то и нечего его взрывать
                if (!SomethingToFlatten)
                {
                    //Удаляем что мы навзрывали - объекты не нужны на чертеже
                    foreach (DBObject obj in objs)
                    {
                        Entity entExplode = obj as Entity;
                        entExplode.Erase();
                    }
                    //Соответственно, если были неплоские примитивы, то результаты взрыва и расплющивания блока
                    //остаются на чертеже
                }
                else if (eraseOrig)
                //Проверим, если нужно - удалим исходный блок
                {
                    ent.UpgradeOpen();//открываем блок на запись
                    //и удаляем
                    ent.Erase();
                };

                // And then we commit
                tr.Commit();

                //Возвращаем значение (было что плющить)
                return SomethingToFlatten;
            }
        }

        private bool FlattenHatch(ObjectId hatchId)
        {
            /*
             * Функция преобразования координат контура штриховки
             * Последовательно пробегает по каждому из контуров штриховки
             * далее последовательно пробегает по каждой из вершин данного контура
             * и зануляет координату Z этой вершины
            */

            //Исходный код для AutoCAD .Net
            //http://forums.autodesk.com/t5/NET/Restore-hatch-boundaries-if-they-have-been-lost-with-NET/m-p/3779514#M33429

            bool result = false;


            using (Transaction tr = acCurDoc.TransactionManager.StartTransaction())
            {
                Hatch hatch = tr.GetObject(hatchId, OpenMode.ForRead) as Hatch;
                if (hatch != null)
                {
                    BlockTableRecord btr = tr.GetObject(hatch.OwnerId, OpenMode.ForWrite) as BlockTableRecord;
                    if (btr != null)
                    {
                        Plane plane = hatch.GetPlane();
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
                                    //Создаем полилиню в текущем пространстве
                                    ObjectId polyId = btr.AppendEntity(poly);
                                    tr.AddNewlyCreatedDBObject(poly, true);

                                    //Плющим полученный контур штриховки
                                    if (flattenPolyline(poly as Entity))
                                    {
                                        //Создание штриховки: http://adndevblog.typepad.com/autocad/2012/07/hatch-using-the-autocad-net-api.html#sthash.ed0Ms37Y.dpuf
                                        ObjectIdCollection ObjIds = new ObjectIdCollection();
                                        ObjIds.Add(polyId);

                                        //Задаем на всякий случай штриховке Elevation =0;
                                        hatch.Elevation = 0;
                                        //Удаляем старый контур
                                        hatch.RemoveLoopAt(i);

                                        //Добавляем новый контур штриховки из сплющенной полилинии
                                        hatch.AppendLoop((int)HatchLoopTypes.Default, ObjIds);
                                        hatch.EvaluateHatch(true);

                                        result = true;
                                    }
                                    else
                                    {
                                        //Ну если полилиния не нуждается в расплющивании, то ее можно удалить.
                                        poly.Erase();
                                    }
                                }
                            }
                            else
                            {//Если не удалось преобразовать контур к полилинии

                                //Выводим сообщение в командную строку
                                cDebug.cDebug.WriteLine("Ошибка обработки: Контур штриховки - не полилиния");
                                //Не будем брать исходный код для штриховок, контур который не сводится к полилинии
                            }
                        }
                    }
                }
                tr.Commit();
            }
            return result;
        }
    }
}

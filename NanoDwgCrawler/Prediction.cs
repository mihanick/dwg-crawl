using DwgDump.MultiContent;
using Multicad;
using Multicad.AplicationServices;
using Multicad.DatabaseServices;
using Multicad.DatabaseServices.StandardObjects;
using Multicad.Dimensions;
using Multicad.Geometry;
using Multicad.Runtime;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

namespace DwgDump
{
	public class Prediction
	{

		private static HttpClient client = null;

		// https://stackoverflow.com/questions/1131425/send-a-file-via-http-post-with-c-sharp
		private static async Task<string> Upload(string actionUrl, string paramString, Stream paramFileStream, byte[] paramFileBytes)
		{
			HttpContent stringContent = new StringContent(paramString);
			// HttpContent fileStreamContent = new StreamContent(paramFileStream);
			HttpContent bytesContent = new ByteArrayContent(paramFileBytes);
			using (var client = new HttpClient())
			using (var formData = new MultipartFormDataContent())
			{
				formData.Add(stringContent, "filename", "filename");
				// formData.Add(fileStreamContent, "file", paramString);
				formData.Add(bytesContent, "file", paramString);
				var response = await client.PostAsync(actionUrl, formData);
				if (!response.IsSuccessStatusCode)
				{
					return null;
				}

				return await response.Content.ReadAsStringAsync();
			}
		}

		/// <summary>
		/// Post an image over predefined serv
		/// </summary>
		/// <returns></returns>
		private static string PostImage(Bitmap image)
		{
			using (MemoryStream stream = new MemoryStream())
			{
				Bitmap blank = new Bitmap(512, 512);
				Graphics g = Graphics.FromImage(blank);
				g.Clear(Color.White);
				g.DrawImage(image, 0, 0);
				// g.DrawImage(DrawArea, 0, 0, DrawArea.Width, DrawArea.Height);

				Bitmap tempImage = new Bitmap(blank);
				blank.Dispose();
				// DrawArea.Dispose();

				//if (extension == ".jpeg")
				//	tempImage.Save(fileName, System.Drawing.Imaging.ImageFormat.Jpeg);
				//else
				//	tempImage.Save(fileName, System.Drawing.Imaging.ImageFormat.Bmp);

				//DrawArea = new Bitmap(tempImage);

				tempImage.Save(stream, ImageFormat.Bmp);
				tempImage.Dispose();

				var t = Task.Run(() => Upload("http://localhost:5000/api/predict", "fragment.bmp", null, stream.ToArray()));
				return t.Result;
			}

			return string.Empty;
		}


		private static void DrawPlineOverBound(PredictionParams param)
		{
			DbPolyline pl = new DbPolyline();
			pl.DbEntity.Color = Color.LightBlue;
			if (param.PredictionClass == 0)
				pl.DbEntity.Color = Color.Blue;
			if (param.PredictionClass < 0)
				pl.DbEntity.Color = Color.Green;

			pl.Polyline = Polyline3d.CreateRectangle(param.Bound.MinPoint, param.Bound.MaxPoint);

			pl.DbEntity.AddToCurrentDocument();
		}

		public class PredictionParams
		{
			public BoundBlock Bound;
			public int PredictionClass;
			public Color Color;
			public double Confidence;

			public PredictionParams()
			{

			}

			/// <summary>
			/// 
			/// </summary>
			/// <param name="boundBase"></param>
			/// <param name="x1">TopLeft X</param>
			/// <param name="y1">TopLeft Y</param>
			/// <param name="x2">Bottom Right X</param>
			/// <param name="y2">Bottom Right Y</param>
			/// <param name="imgSize"></param>
			/// <returns></returns>
			public static PredictionParams FromPixels(BoundBlock boundBase, double x1, double y1, double x2, double y2, int imgSize = 512)
			{
				// Max size of source bound
				var baseSize = Math.Max(boundBase.SizeByX, boundBase.SizeByY);

				// Min point is a point of bound with aspect ratio 1:1
				var minPnt = new Point3d(boundBase.MinPoint.X, boundBase.MaxPoint.Y - baseSize, boundBase.MinPoint.Z);

				var dx2 = x2 * baseSize / imgSize;
				var dy2 = (imgSize - y2) * baseSize / imgSize;

				var dx1 = x1 * baseSize / imgSize;
				var dy1 = (imgSize - y1) * baseSize / imgSize;

				var mindx = minPnt.X + dx1;
				var maxdx = minPnt.X + dx2;

				var mindy = minPnt.Y + dy1;
				var maxdy = minPnt.Y + dy2;

				Point3d minPt = new Point3d(mindx, mindy, minPnt.Z);
				Point3d maxPt = new Point3d(maxdx, maxdy, minPnt.Z);

				var result = new BoundBlock(minPt, maxPt);

				return new PredictionParams()
				{
					Bound = result
				};
			}
		}

		private static Point3d MidPoint(Point3d p1, Point3d p2)
		{
			return new Point3d(
				(p1.X + p2.X) / 2,
				(p1.Y + p2.Y) / 2,
				(p1.Z + p2.Z) / 2
				);
		}

		private static Point3d SnapToNearestEntity(Point3d point, double aperture = 10)
		{
			// TODO: Dynamic aperture
			var appertureOffset = new Vector3d(10 * aperture, 10 * aperture, 0);
			var bound = new BoundBlock(point - appertureOffset, point + appertureOffset);

			ObjectFilter filter = new ObjectFilter();
			filter.SetBound(bound);
			filter.ExcludeHiddenLayers = true;
			filter.ExcludeSystemLayers = true;
			filter.IncludeInvisible = false;

			var objectsInBound = new List<McDbEntity>();
			foreach (var id in filter.GetObjects())
			{
				var dbo = id.GetObject()?.Cast<McDbEntity>();
				if (dbo != null)
				{
					// BUG: These types return same point as nearest
					if (id.GetObject() is McLinearDimension || id.GetObject() is DbText)
						continue;
					objectsInBound.Add(dbo);
				}
			}

			double closestDist = double.PositiveInfinity;
			Point3d adjustedPoint = point;
			foreach (var dbo in objectsInBound)
			{
				var entPnt = dbo.GetNearestPoint(point);

				// Only look at points within aperture
				if (!bound.Contains(entPnt))
					continue;

				var dist = (point - entPnt).Length;
				if (dist < closestDist)
				{
#if DEBUG
					if (dist == 0)
						dbo.Color = Color.RosyBrown;
#endif
					closestDist = dist;
					adjustedPoint = entPnt;
				}

			}

			// DrawPlineOverBound(new PredictionParams() { Bound = bound });
			return adjustedPoint;
		}

		[CommandMethod("redd", CommandFlags.NoCheck | CommandFlags.NoPrefix | CommandFlags.Redraw)]
		public static void Predict()
		{
			List<McObjectId> currentSelection = new List<McObjectId>(McObjectManager.SelectionSet.CurrentSelection);
			if (currentSelection.Count == 0)
				currentSelection = McObjectManager.SelectObjects("Select fragment").ToList();

			// write image
			(var image, var bound) = BmpFromDwg.CreatePreview(currentSelection);

			string json = PostImage(image);

			// parse response
			// Parse json of token request
			object ret = JsonConvert.DeserializeObject(json);
			JArray jObj = (JArray)ret;

			PredictionParams allEntitiesBound = new PredictionParams()
			{
				Bound = bound,
				PredictionClass = -1
			};

			// DEBUG: Draw source bound from which we created an image to recognize
			// DrawPlineOverBound(allEntitiesBound);

			List<PredictionParams> predictions = new List<PredictionParams>();
			foreach (var predLine in jObj[0])
			{
				double.TryParse(predLine[0].ToString(), out var cx);
				double.TryParse(predLine[1].ToString(), out var cy);
				double.TryParse(predLine[2].ToString(), out var h);
				double.TryParse(predLine[3].ToString(), out var w);
				double.TryParse(predLine[4].ToString(), out var conf);
				double.TryParse(predLine[5].ToString(), out var cls);

				// McContext.ShowNotification(string.Format("{0} {1} {2} {3} {4} {5}", cx, cy, h, w, conf,cls));
				var bb = PredictionParams.FromPixels(bound, cx, cy, h, w, 640);
				bb.PredictionClass = (int)cls;
				bb.Confidence = conf;

				predictions.Add(bb);
			}

			// decide which bound and how to draw
			// we'll do that by choosing relative position of center of bound box line
			// in respect to base bound center
			foreach (var param in predictions)
			{
				List<McLinearDimension> dims = new List<McLinearDimension>();
				if (param.PredictionClass == 0)
				{
					var pnts = param.Bound.Get2dCorners();
					var relBoundVec = (allEntitiesBound.Bound.Center - param.Bound.Center);
					var angle45 = new Vector3d(1, 1, 0);

					var relativeAngle = relBoundVec.GetAngleTo(angle45);
					var relDirection = relBoundVec.CrossProduct(angle45);

					Point3d p1, p2, p3;
					if (relDirection.Z > 0 && relativeAngle <= Math.PI / 2)
					{
						p1 = pnts[0];
						p2 = pnts[1];
						p3 = MidPoint(pnts[2], pnts[3]);
					}
					else if (relDirection.Z > 0 && relativeAngle >= Math.PI / 2)
					{
						p1 = pnts[1];
						p2 = pnts[2];
						p3 = MidPoint(pnts[0], pnts[3]);
					}
					else if (relDirection.Z < 0 && relativeAngle <= Math.PI / 2)
					{
						p1 = pnts[3];
						p2 = pnts[2];
						p3 = MidPoint(pnts[0], pnts[1]);
					}
					else // (relDirection.Z < 0 && relativeAngle >= Math.PI / 2)
					{
						p1 = pnts[0];
						p2 = pnts[3];
						p3 = MidPoint(pnts[1], pnts[2]);
					}

					// draw dimensions over prediction
					McLinearDimension dim = new McLinearDimension();
					p1 = SnapToNearestEntity(p1);
					p2 = SnapToNearestEntity(p2);
					dim.SetPosition(0, p1);
					dim.SetPosition(1, p2);
					// dim.SetPosition(2, p3);
					// dim.TextPosition = p3;
					dim.Suffix = string.Format("[@{0:0.}%]", 100 * param.Confidence);
					dim.Direction = p2 - p1;
					dim.LinePosition = p3;
					dims.Add(dim);
				}
				else
					continue;
				// DrawPlineOverBound(param);

				foreach (McLinearDimension dim in dims)
				{
					dim.DbEntity.Color = Color.DarkOrange;
					dim.DbEntity.AddToCurrentDocument();
				}
			}

		}
	}
}
